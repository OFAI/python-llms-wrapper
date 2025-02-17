"""
Module related to using LLMs.
"""
import os
import warnings
# TODO: Remove after https://github.com/BerriAI/litellm/issues/7560 is fixed
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._config")
import litellm
import time
import inspect
import docstring_parser
from loguru import logger
from typing import Optional, Dict, List, Union, Tuple, Callable
from copy import deepcopy

from litellm import completion, completion_cost
from litellm.utils import get_model_info, get_supported_openai_params, supports_response_schema
from litellm.utils import supports_function_calling, supports_parallel_function_calling
from llms_wrapper.utils import dict_except

ROLES = ["user", "assistant", "system"]


class LLMS:
    """
    Class that represents a preconfigured set of large language modelservices.
    """

    def __init__(self, config: Dict, debug: bool = False):
        """
        Initialize the LLMS object with the given configuration.
        """
        self.config = deepcopy(config)
        self.debug = debug
        # convert the config into a dictionary of LLM objects where the key is the alias of the LLM
        self.llms: Dict[str, "LLM"] = {}
        for llm in self.config["llms"]:
            if not isinstance(llm, dict):
                raise ValueError(f"Error: LLM entry is not a dict: {llm}")
            alias = llm["alias"]
            if alias in self.llms:
                raise ValueError(f"Error: Duplicate LLM alis {alias} in configuration")
            # make a copy
            llmdict = deepcopy(llm)
            llmdict["_cost"] = 0
            llmdict["_elapsed_time"] = 0
            llm = LLM(llmdict, self)
            self.llms[alias] = llm

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

    def cost_per_token(self, llmalias: str) -> Tuple[float, float]:
        """
        Return the estimated cost per prompt and completion token for the given model.
        This may be wrong or cost may get calculated in a different way, e.g. depending on
        cache, response time etc.
        """
        return litellm.cost_per_token(self.llms[llmalias]["llm"], prompt_tokens=1, completion_tokens=1)

    def max_output_tokens(self, llmalias: str) -> int:
        """
        Return the maximum number of prompt tokens that can be sent to the model.
        """
        return litellm.get_max_tokens(self.llms[llmalias]["llm"])

    def max_input_tokens(self, llmalias: str) -> Optional[int]:
        """
        Return the maximum number of tokens possible in the prompt or None if not known.
        """
        try:
            info = get_model_info(self.llms[llmalias]["llm"])
            return info["max_input_tokens"]
        except:
            # the model is not mapped yet, return None to indicate we do not know
            return None

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
        return "response_format" in params

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
    ) -> Dict[str, any]:
        """
        Query the specified LLM with the given messages.

        Args:
            llmalias: the alias/name of the LLM to query
            messages: a list of message dictionaries with role and content keys
            debug: if True, debug logging is enabled

        Returns:
            A dictionary with keys answer and error and optionally cost-related keys and optionally
                the full original response. If there is an error, answer is the empty string and error contains the error,
                otherwise answer contains the response and error is the empty string.
                The boolean key "ok" is True if there is no error, False otherwise.
        """
        if self.debug:
            debug = True
        if debug:
            #  litellm.set_verbose = True    ## deprecated!
            os.environ['LITELLM_LOG'] = 'DEBUG'
        llm = self.llms[llmalias]
        if not messages:
            raise ValueError(f"Error: No messages to send to the LLM: {llmalias}, messages: {messages}")
        if debug:
            logger.debug(f"Sending messages to {llmalias}: {messages}")
        # prepare the keyword arguments for colling completion
        completion_kwargs = dict_except(
            llm,
            [
                "llm", "alias", "api_key", "api_url", "user", "password",
                "api_key_env", "user_env", "password_env", "_cost", "_elapsed_time"])
        error = None
        if llm.get("api_key"):
            completion_kwargs["api_key"] = llm["api_key"]
        elif llm.get("api_key_env"):
            completion_kwargs["api_key"] = os.getenv(llm["api_key_env"])
        if llm.get("api_url"):
            completion_kwargs["api_base"] = llm["api_url"]
        if tools is not None:
            completion_kwargs["tools"] = tools
        ret = {}
        if debug:
            logger.debug(f"Calling completion with {completion_kwargs}")
        try:
            start = time.time()
            response = completion(
                model=llm["llm"],
                messages=messages,
                **completion_kwargs)
            elapsed = time.time() - start
            logger.debug(f"Full Response: {response}")
            llm["_elapsed_time"] += elapsed
            ret["elapsed_time"] = elapsed
            if return_response:
                ret["response"] = response
                ret["kwargs"] = completion_kwargs
            if return_cost:
                ret["cost"] = completion_cost(
                    completion_response=response,
                    model=llm["llm"],
                    messages=messages,
                )
                llm["_cost"] += ret["cost"]
                usage = response['usage']
                logger.debug(f"Usage: {usage}")
                ret["n_completion_tokens"] = usage.completion_tokens
                ret["n_prompt_tokens"] = usage.prompt_tokens
                ret["n_total_tokens"] = usage.total_tokens
            response_message = response['choices'][0]['message']
            ret["answer"] = response_message['content']
            ret["error"] = ""
            ret["ok"] = True
            # TODO: if feasable handle all tool calling here or in a separate method which does
            #   all the tool calling steps (up to a specified maximum).
            if tools is not None:
                ret["tool_calls"] = response_message.tool_calls
                ret["response_message"] = response_message
        except Exception as e:
            ret["error"] = str(e)
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
    ) -> Dict[str, any]:
        llmalias = self.config["alias"]
        return self.llmsobject.query(
            llmalias,
            messages=messages,
            tools=tools,
            return_cost=return_cost,
            return_response=return_response,
            debug=debug)

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

