"""
Module related to using LLMs.
"""
import os
import warnings
# TODO: Remove after https://github.com/BerriAI/litellm/issues/7560 is fixed
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._config")
import litellm
import time
import traceback
import inspect
import docstring_parser
from loguru import logger
from typing import Optional, Dict, List, Union, Tuple, Callable
from copy import deepcopy

from litellm import completion, completion_cost
from litellm.utils import get_model_info, get_supported_openai_params, supports_response_schema
from litellm.utils import supports_function_calling, supports_parallel_function_calling
from llms_wrapper.utils import dict_except
from llms_wrapper.model_list import model_list

# roles to consider in messages for replacing variables in the content
ROLES = ["user", "assistant", "system"]

# fields in the config to NOT pass on to the LLM
KNOWN_LLM_CONFIG_FIELDS = [
    "llm", "alias",
    "api_key",   # we pass this on separately, if necessary
    "api_url",
    "user",
    "password",
    "api_key_env",
    "user_env",
    "password_env",
    "api_key_env", "user_env", "password_env",
    "cost_per_prompt_token",
    "cost_per_output_token",
    "max_output_tokens",
    "max_input_tokens",
    "use_phoenix",
]


class LLMS:
    """
    Class that represents a preconfigured set of large language modelservices.
    """

    def __init__(self, config: Dict = None, debug: bool = False, use_phoenix: Optional[Union[str | Tuple[str, str]]] = None):
        """
        Initialize the LLMS object with the given configuration.

        Use phoenix is either None or the URI of the phoenix endpoing or a tuple with the URI and the
        project name (so far this only works for local phoenix instances). Default URI for a local installation
        is "http://0.0.0.0:6006/v1/traces"
        """
        if config is None:
            config = dict(llms=[])
        self.config = deepcopy(config)
        self.debug = debug
        if not use_phoenix and config.get("use_phoenix"):
            use_phoenix = config["use_phoenix"]
        if use_phoenix:
            if isinstance(use_phoenix, str):
                use_phoenix = (use_phoenix, "default")
                print("importing")
            from phoenix.otel import register
            from openinference.instrumentation.litellm import LiteLLMInstrumentor
            # register
            tracer_provider = register(
                project_name=use_phoenix[1],  # Default is 'default'
                # auto_instrument=True,  # Auto-instrument your app based on installed OI dependencies
                endpoint=use_phoenix[0],
            )
            # instrument
            LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
        # convert the config into a dictionary of LLM objects where the key is the alias of the LLM
        self.llms: Dict[str, "LLM"] = {}
        for llm in self.config["llms"]:
            if not isinstance(llm, dict):
                raise ValueError(f"Error: LLM entry is not a dict: {llm}")
            alias = llm.get("alias", llm["llm"])
            if alias in self.llms:
                raise ValueError(f"Error: Duplicate LLM alis {alias} in configuration")
            llmdict = deepcopy(llm)
            llmdict["_cost"] = 0
            llmdict["_elapsed_time"] = 0
            llm = LLM(llmdict, self)
            self.llms[alias] = llm

    def known_models(self, provider=None) -> List[str]:
        """
        Get a list of known models.
        """
        return model_list(provider)

    def list_models(self) -> List["LLM"]:
        """
        Get a list of model configuration objects
        """
        return [llm for llm in self.llms.values()]

    def list_aliases(self) -> List[str]:
        """
        List the (unique) alias names in the configuration.
        """
        return list(self.llms.keys())

    def get(self, alias: str) -> Optional[Dict]:
        """
        Get the LLM configuration object with the given alias.
        """
        return self.llms.get(alias, None)

    def __getitem__(self, item: str) -> "LLM":
        """
        Get the LLM configuration object with the given alias.
        """
        return self.llms[item]

    def elapsed(self, llmalias: Union[str, List[str], None] = None):
        """
        Return the elapsed time so far for the given llm alias given list of llm aliases
        or all llms if llmalias is None. Elapsed time is only accumulated for invocations of
        the query method with return_cost=True.
        """
        if llmalias is None:
            return sum([llm["_elapsed_time"] for llm in self.llms.values()])
        if isinstance(llmalias, str):
            return self.llms[llmalias]["_elapsed_time"]
        return sum([self.llms[alias]["_elapsed_time"] for alias in llmalias])
    
    def get_llm_info(self, llmalias: str, name: str) -> any:
        """
        For convenience, any parameter with a name staring with an underscore can be used to configure 
        our own properties of the LLM object. This method returns the value of the given parameter name of None
        if not defined, where the name should not include the leading underscore.
        """
        return self.llms[llmalias].config.get("_"+name, None)
    
    def default_max_tokens(self, llmalias: str) -> int:
        """
        Return the default maximum number of tokens that the LLM will produce. This is sometimes smaller thant the actual
        max_tokens, but not supported by LiteLLM, so we use whatever is configured in the config and fall back
        to the actual max_tokens if not defined.
        """
        ret = self.llms[llmalias].config.get("default_max_tokens")
        if ret is None:
            ret = self.max_output_tokens(llmalias)
        return ret
    

    def cost(self, llmalias: Union[str, List[str], None] = None):
        """
        Return the cost accumulated so far for the given llm alias given list of llm aliases
        or all llms if llmalias is None. Costs are only accumulated for invocations of
        the query method with return_cost=True.
        """
        if llmalias is None:
            return sum([llm["_cost"] for llm in self.llms.values()])
        if isinstance(llmalias, str):
            return self.llms[llmalias]["_cost"]
        return sum([self.llms[alias]["_cost"] for alias in llmalias])

    def cost_per_token(self, llmalias: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Return the estimated cost per prompt and completion token for the given model.
        This may be wrong or cost may get calculated in a different way, e.g. depending on
        cache, response time etc.
        If the model is not in the configuration, this makes and attempt to just get the cost as 
        defined by the LiteLLM backend.
        If no cost is known this returns 0.0, 0.0
        """
        llm = self.llms.get(llmalias)
        cc, cp = None, None
        if llm is not None:
            cc = llm.get("cost_per_prompt_token")
            cp = llm.get("cost_per_completion_token")
            llmname = llm["llm"]
        else:
            llmname = llmalias
        if cc is None or cp is None:
            try:
                tmpcp, tmpcc = litellm.cost_per_token(llmname, prompt_tokens=1, completion_tokens=1)
            except:
                tmpcp, tmpcc = None, None
            if cc is None:
                cc = tmpcc
            if cp is None:
                cp = tmpcp
        return cc, cp

    def max_output_tokens(self, llmalias: str) -> Optional[int]:
        """
        Return the maximum number of prompt tokens that can be sent to the model.
        """
        llm = self.llms.get(llmalias)
        ret = None
        if llm is not None:
            llmname = llm["llm"]
            ret = llm.get("max_output_tokens")
        else:
            llmname = llmalias
        if ret is None:
            try:
                # ret = litellm.get_max_tokens(self.llms[llmalias]["llm"])
                info = get_model_info(llmname)
                ret = info.get("max_output_tokens")
            except:
                ret = None
        return ret

    def max_input_tokens(self, llmalias: str) -> Optional[int]:
        """
        Return the maximum number of tokens possible in the prompt or None if not known.
        """
        llm = self.llms.get(llmalias)
        ret = None
        if llm is not None:
            ret = llm.get("max_input_tokens")
            llmname = llm["llm"]
        else:
            llmname = llmalias
        if ret is None:
            try:
                info = get_model_info(llmname)
                ret = info.get("max_input_tokens")
            except:
                ret = None
        return ret

    def set_model_attributes(
            self, llmalias: str,
            input_cost_per_token: float,
            output_cost_per_token: float,
            input_cost_per_second: float,
            max_prompt_tokens: int,
    ):
        """
        Set or override the attributes for the given model.

        NOTE: instead of using this method, the same parameters can alos
        be set in the configuration file to be passed to the model invocation call.
        """
        llmname = self.llms[llmalias]["llm"]
        provider, model = llmname.split("/", 1)
        litellm.register_model(
            {
                model: {
                    "max_tokens": max_prompt_tokens,
                    "output_cost_per_token": output_cost_per_token,
                    "input_cost_per_token": input_cost_per_token,
                    "input_cost_per_second": input_cost_per_second,
                    "litellm_provider": provider,
                    "mode": "chat",
                }
            }
        )

    @staticmethod
    def make_messages(
            query: Optional[str] = None,
            prompt: Optional[Dict[str, str]] = None,
            messages: Optional[List[Dict[str, str]]] = None,
            keep_n: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Construct updated messages from the query and/or prompt data.

        Args:
            query: A query text, if no prompt is given, a message with this text for role user is created.
            prompt: a dict mapping roles to text templates, where the text template may contain the string "${query}"
            messages: previous messages to include in the new messages
            keep_n: the number of messages to keep, if None, all messages are kept, otherwise the first message and
                the last keep_n-1 messages are kept.

        Returns:
            A list of message dictionaries
        """
        if messages is None:
            messages = []
        if query is None and prompt is None:
            raise ValueError("Error: Both query and prompt are None")
        if query is None:
            # convert the prompt as is to messages
            for role, content in prompt.items():
                if content and role in ROLES:
                    messages.append(dict(role=role, content=content))
        elif prompt is None:
            messages.append({"content": query, "role": "user"})
        else:
            for role, content in prompt.items():
                if content and role in ROLES:
                    messages.append(dict(role=role, content=content.replace("${query}", query)))
        # if we have more than keep_n messages, remove oldest message but the first so that we have keep_n messages
        if keep_n is not None and len(messages) > keep_n:
            messages = messages[:1] + messages[-keep_n:]
        return messages

    @staticmethod
    def make_tooling(functions: Union[Callable, List[Callable]]) -> List[Dict]:
        """
        Automatically create the tooling descriptions for a function or list of functions, based on the
        function(s) documentation strings.
        """
        if not isinstance(functions, list):
            functions = [functions]
        tools = []
        for func in functions:
            if not callable(func):
                raise ValueError(f"Error: {func} is not callable")
            doc = docstring_parser.parse(func.__doc__)
            argspec = inspect.getfullargspec(func)
            nrequired = len(argspec.args) - len(argspec.defaults) if argspec.defaults else len(argspec.args)
            # for each parameter get the type as specified in the docstring, if not specified, get the
            # name of the type from the argspec annotation information, if not specified there, assume string
            argtypes = []
            for idx, aname in enumerate(argspec.args):
                if idx < len(doc.params):
                    argtypes.append(doc.params[idx].type_name)
                # it seems proper python types are not supported?
                # elif argspec.annotations.get(aname):
                #     argtypes.append(argspec.annotations[aname].__name__)
                else:
                    argtypes.append("string")
            argdescs = []
            for idx, aname in enumerate(argspec.args):
                if idx < len(doc.params):
                    argdescs.append(doc.params[idx].description)
                else:
                    raise ValueError(f"Error: Missing description for parameter {aname} in doc of function {func.__name__}")
            desc = doc.short_description + "\n\n" + doc.long_description
            tools.append({
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            doc.params[i].arg_name: {
                                "type": argtypes[i],
                                "description": argdescs[i],
                            } for i in range(len(argspec.args))
                        },
                        "required": [param.arg_name for param in doc.params[:nrequired]],
                    },
                },
            })
        return tools

    def supports_response_format(self, llmalias: str) -> bool:
        """
        Check if the model supports the response format parameters. This usually just indicates support
        for response_format "json".
        """
        params = get_supported_openai_params(model=self.llms[llmalias]["llm"],
                                             custom_llm_provider=self.llms[llmalias].get("custom_provider"))
        ret = "response_format" in params
        return ret

    def supports_json_schema(self, llmalias: str) -> bool:
        """
        Check if the model supports the json_schema parameter
        """
        return supports_response_schema(model=self.llms[llmalias]["llm"],
                                        custom_llm_provider=self.llms[llmalias].get("custom_provider"))

    def supports_function_calling(self, llmalias: str, parallel=False) -> bool:
        """
        Check if the model supports function calling
        """
        if parallel:
            return supports_parallel_function_calling(
                model=self.llms[llmalias]["llm"],
                )
        return supports_function_calling(
            model=self.llms[llmalias]["llm"],
            custom_llm_provider=self.llms[llmalias].get("custom_provider"))

    def query(
            self,
            llmalias: str,
            messages: List[Dict[str, str]],
            tools: Optional[List[Dict]] = None,
            return_cost: bool = False,
            return_response: bool = False,
            debug=False,
            litellm_debug=None,
            **kwargs,
    ) -> Dict[str, any]:
        """
        Query the specified LLM with the given messages.

        Args:
            llmalias: the alias/name of the LLM to query
            messages: a list of message dictionaries with role and content keys
            tools: TBD
            return_cost: whether or not LLM invocation costs should get returned
            return_response: whether or not the complete reponse should get returned
            debug: if True, debug logging is enabled
            litellm_debug: if True, litellm debug logging is enabled, if False, disabled, if None, use debug setting
            kwargs: any additional keyword arguments to pass on to the LLM 

        Returns:
            A dictionary with keys answer and error and optionally cost-related keys and optionally
                the full original response. If there is an error, answer is the empty string and error contains the error,
                otherwise answer contains the response and error is the empty string.
                The boolean key "ok" is True if there is no error, False otherwise.
        """
        if self.debug:
            debug = True
        if litellm_debug is None and debug or litellm_debug:
            #  litellm.set_verbose = True    ## deprecated!
            os.environ['LITELLM_LOG'] = 'DEBUG'
        llm = self.llms[llmalias].config
        if not messages:
            raise ValueError(f"Error: No messages to send to the LLM: {llmalias}, messages: {messages}")
        if debug:
            logger.debug(f"Sending messages to {llmalias}: {messages}")
        # prepare the keyword arguments for colling completion
        completion_kwargs = dict_except(
            llm,
            KNOWN_LLM_CONFIG_FIELDS,
            ignore_underscored=True,
        )
        if llm.get("api_key"):
            completion_kwargs["api_key"] = llm["api_key"]
        elif llm.get("api_key_env"):
            completion_kwargs["api_key"] = os.getenv(llm["api_key_env"])
        if llm.get("api_url"):
            completion_kwargs["api_base"] = llm["api_url"]
        if tools is not None:
            completion_kwargs["tools"] = tools
        ret = {}
        if kwargs:
            completion_kwargs.update(kwargs)
        if debug:
            logger.debug(f"Calling completion with kwargs {completion_kwargs}")
        try:
            start = time.time()
            response = litellm.completion(
                model=llm["llm"],
                messages=messages,
                **completion_kwargs)
            elapsed = time.time() - start
            logger.debug(f"Full Response: {response}")
            llm["_elapsed_time"] += elapsed
            ret["elapsed_time"] = elapsed
            if return_response:
                ret["response"] = response
                # prevent the api key from leaking out
                if "api_key" in completion_kwargs:
                    del completion_kwargs["api_key"]
                ret["kwargs"] = completion_kwargs
            if return_cost:
                try:
                    ret["cost"] = completion_cost(
                        completion_response=response,
                        model=llm["llm"],
                        messages=messages,
                    )
                except Exception as e:
                    logger.debug(f"Error in completion_cost for model {llm['llm']}: {e}")
                    ret["cost"] = 0.0
                llm["_cost"] += ret["cost"]
                usage = response['usage']
                logger.debug(f"Usage: {usage}")
                ret["n_completion_tokens"] = usage.completion_tokens
                ret["n_prompt_tokens"] = usage.prompt_tokens
                ret["n_total_tokens"] = usage.total_tokens
            response_message = response['choices'][0]['message']
            # Does not seem to work see https://github.com/BerriAI/litellm/issues/389
            # ret["response_ms"] = response["response_ms"]
            ret["finish_reason"] = response['choices'][0].get('finish_reason', "UNKNOWN")
            ret["answer"] = response_message['content']
            ret["error"] = ""
            ret["ok"] = True
            # TODO: if feasable handle all tool calling here or in a separate method which does
            #   all the tool calling steps (up to a specified maximum).
            if tools is not None:
                ret["tool_calls"] = response_message.tool_calls
                ret["response_message"] = response_message
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            filename, lineno, funcname, text = tb[-1]
            ret["error"] = str(e) + f" in {filename}:{lineno} {funcname}"
            if debug:
                logger.error(f"Returning error: {e}")
            ret["answer"] = ""
            ret["ok"] = False
        return ret


# For now, this class simply represents the LLM by the config dict and a pointer to the LLMS object it is contained
# in. In order to avoid changing any code in the LLMS object where we expect the llm config to be a dict
# we also implement the __getitem__, __setitem__, and get methods to access the nested dict in the llm object.
class LLM:
    def __init__(self, config: Dict, llmsobject: LLMS):
        self.config = config
        self.llmsobject = llmsobject

    def __getitem__(self, item: str) -> any:
        return self.config[item]

    def __setitem__(self, key: str, value: any):
        self.config[key] = value

    def get(self, item: str, default=None) -> any:
        return self.config.get(item, default)

    def items(self):
        return self.config.items()

    def query(
            self,
            messages: List[Dict[str, str]],
            tools: Optional[List[Dict]] = None,
            return_cost: bool = False,
            return_response: bool = False,
            debug=False,
            **kwargs,
    ) -> Dict[str, any]:
        llmalias = self.config["alias"]
        return self.llmsobject.query(
            llmalias,
            messages=messages,
            tools=tools,
            return_cost=return_cost,
            return_response=return_response,
            debug=debug, **kwargs)

    def __str__(self):
        return f"LLM({self.config['alias']})"

    def __repr__(self):
        return f"LLM({self.config['alias']})"

    # other methods which get delegated to the parent LLMS object
    def make_messages(self, query: Optional[str] = None, prompt: Optional[Dict[str, str]] = None,
                      messages: Optional[List[Dict[str, str]]] = None, keep_n: Optional[int] = None) -> List[Dict[str, str]]:
        return self.llmsobject.make_messages(query, prompt, messages, keep_n)

    def cost_per_token(self) -> Tuple[float, float]:
        return self.llmsobject.cost_per_token(self.config["alias"])

    def max_output_tokens(self) -> int:
        return self.llmsobject.max_output_tokens(self.config["alias"])

    def max_input_tokens(self) -> Optional[int]:
        return self.llmsobject.max_input_tokens(self.config["alias"])

    def set_model_attributes(self, input_cost_per_token: float, output_cost_per_token: float,
                             input_cost_per_second: float, max_prompt_tokens: int):
        return self.llmsobject.set_model_attributes(self.config["alias"], input_cost_per_token, output_cost_per_token,
                                                   input_cost_per_second, max_prompt_tokens)

    def elapsed(self):
        return self.llmsobject.elapsed(self.config["alias"])

    def cost(self):
        return self.llmsobject.cost(self.config["alias"])

    def supports_response_format(self) -> bool:
        return self.llmsobject.supports_response_format(self.config["alias"])

    def supports_json_schema(self) -> bool:
        return self.llmsobject.supports_json_schema(self.config["alias"])

    def supports_function_calling(self, parallel=False) -> bool:
        return self.llmsobject.supports_function_calling(self.config["alias"], parallel)

