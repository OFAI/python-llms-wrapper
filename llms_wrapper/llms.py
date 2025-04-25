"""
Module related to using LLMs.
"""
import os
import warnings
# TODO: Remove after https://github.com/BerriAI/litellm/issues/7560 is fixed
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._config")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
import litellm
import json
import time
import traceback
import inspect
import docstring_parser
from loguru import logger
import typing
from typing import Optional, Dict, List, Union, Tuple, Callable, get_args, get_origin
from copy import deepcopy

from litellm import completion, completion_cost, token_counter
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
    "min_delay",   # minimum delay between queries for that model
]


def toolnames2funcs(tools):
    """
    Convert a list of tool names to a dictionary of functions.
    
    Args:
        tools: List of tools, each with a name.
    
    Returns:
        Dictionary of function names to functions.

    Raises:
        Exception: If a function is not found.
    """
    fmap = {}
    for tool in tools:
        name = tool["function"]["name"]
        func = get_func_by_name(name)
        if func is None:
            raise Exception(f"Function {name} not found")
        fmap[name] = func
    return fmap


def get_func_by_name(name):
    """
    Get a function by name.
    
    Args:
        name: Name of the function.
    
    Returns:
        Function if found, None otherwise.
    
    Raises:
        Exception: If a function is not found.
    """
    for frame_info in inspect.stack():
        frame = frame_info.frame
        func = frame.f_locals.get(name) or frame.f_globals.get(name)
        if callable(func):
            return func
    return None  # Not found

def ptype2schema(py_type):
    """
    Convert a Python type to a JSON schema.
    
    Args:
        py_type: Python type to convert.
    
    Returns:
        JSON schema for the given Python type.
    
    Raises:
        ValueError: If the type is not supported.
    """ 
    # Handle bare None
    if py_type is type(None):
        return {"type": "null"}

    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is None:
        # Base types
        if py_type is str:
            return {"type": "string"}
        elif py_type is int:
            return {"type": "integer"}
        elif py_type is float:
            return {"type": "number"}
        elif py_type is bool:
            return {"type": "boolean"}
        elif py_type is type(None):
            return {"type": "null"}
        else:
            return {"type": "string"}  # Fallback

    elif origin is list or origin is typing.List:
        item_type = ptype2schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_type}

    elif origin is dict or origin is typing.Dict:
        key_type, val_type = args if args else (str, str)
        # JSON Schema requires string keys
        if key_type != str:
            raise ValueError("JSON object keys must be strings")
        return {"type": "object", "additionalProperties": ptype2schema(val_type)}

    elif origin is typing.Union:
        # Flatten nested Union
        flat_args = []
        for arg in args:
            if get_origin(arg) is typing.Union:
                flat_args.extend(get_args(arg))
            else:
                flat_args.append(arg)

        schemas = [ptype2schema(a) for a in flat_args]
        return {"anyOf": schemas}

    elif origin is typing.Literal:
        return {"enum": list(args)}

    else:
        return {"type": "string"}  # fallback for unsupported/unknown
    
def function2schema(func, include_return_type=True):
    """
    Convert a function to a JSON schema.
    
    Args:
        func: Function to convert.
        include_return_type: Whether to include the return type in the schema.
    
    Returns:
        JSON schema for the given function.
    
    Raises:
        ValueError: If the function docstring is empty.
    """ 
    doc = docstring_parser.parse(func.__doc__)
    desc = doc.short_description + "\n\n" + doc.long_description if doc.long_description else doc.short_description
    if not desc:
        raise ValueError("Function docstring is empty")
    argdescs = {arg.arg_name: arg.description for arg in doc.params}    
    argtypes = {}
    for arg in doc.params:
        argtype = arg.type_name
        # if the argtype is not specified, skip, we will use the argument type
        if argtype is None:
            continue
        # if the argtype starts with a brace, we assume it is already specified as a JSON schema
        if argtype.startswith("{"):
            argtypes[arg.arg_name] = json.loads(argtype)
        else:
            # otherwise, we assume it is a python type            
            argtypes[arg.arg_name] = ptype2schema(argtype)
    retdesc = doc.returns.description if doc.returns else ""
    if not retdesc:
        raise ValueError("Function return type is not specified in docstring")
    retschema = ptype2schema(func.__annotations__.get("return", None))
    desc = desc + "\n\n" + "The function returns: " + str(retdesc)
    if include_return_type:
        desc = desc + "\n\n" + "The return type is: " + str(retschema)
    sig = inspect.signature(func)
    parameters = sig.parameters

    props = {}
    required = []

    for name, param in parameters.items():
        if name == 'self':
            continue

        if name in argtypes:
            schema = argtypes[name]
        else:
            # Use the type annotation if available, otherwise default to string
            ptype = param.annotation if param.annotation != inspect.Parameter.empty else str
            schema = ptype2schema(ptype)
        schema["description"] = argdescs.get(name, "")

        if param.default != inspect.Parameter.empty:
            schema["default"] = param.default
        else:
            required.append(name)

        props[name] = schema

    funcschema = {
        "name": func.__name__,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": props,
            "required": required
        }
    }
    toolschema = dict(type="function", function=funcschema)
    return toolschema


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
            llmdict["_last_request_time"] = 0
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
            message: Optional[Dict[str, str]] = None,
            messages: Optional[List[Dict[str, str]]] = None,
            keep_n: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Construct updated messages from the query and/or prompt data.

        Args:
            query: A query text, if no prompt or message is given, a message with this text for role user is created.
                Otherwise, this string is used to replace "${query}" in the prompt content if that is a string.
            prompt: a dict mapping roles to text templates, where the text template may contain the string "${query}"
                A prompt looks like this: {"user": "What is the capital of ${query}?", "system": "Be totally helpful!"}
                If prompt is specified, the query string is used to replace "${query}" in the prompt content.
            message: a message to use as is; if messages is given, this message is added to messages and the
                combination is sent to the LLM, if messages is None, this message is sent as is.
                A message looks like this: {"role": "user", "content": "What is the capital of France?"}
                If message is specified, query and prompt are ignored.
            messages: previous messages to include in the new messages
            keep_n: the number of messages to keep, if None, all messages are kept, otherwise the first message and
                the last keep_n-1 messages are kept.

        Returns:
            A list of message dictionaries
        """
        if messages is None:
            messages = []
        if query is None and prompt is None and message is None:
            raise ValueError("Error: All of query and prompt and message are None")
        if message is not None:
            messages.append(message)
            return messages
        elif prompt is not None:
            if query:
                for role, content in prompt.items():
                    if content and role in ROLES:
                        messages.append(dict(role=role, content=content.replace("${query}", query)))
            else:
                # convert the prompt as is to messages
                for role, content in prompt.items():
                    if content and role in ROLES:
                        messages.append(dict(role=role, content=content))
        else:
            messages.append({"content": query, "role": "user"})
        # if we have more than keep_n messages, remove oldest message but the first so that we have keep_n messages
        if keep_n is not None and len(messages) > keep_n:
            messages = messages[:1] + messages[-keep_n:]
        return messages

    @staticmethod
    def make_tooling(functions: Union[Callable, List[Callable]]) -> List[Dict]:
        """
        Automatically create the tooling descriptions for a function or list of functions, based on the
        function(s) documentation strings.

        The documentation string for each of the functions should be in in a format supported
        by the docstring_parser package (Google, Numpy, or ReST).

        The description of the function should be given in detail and in a way that will
        be useful to the LLM. The same goes for the description of each of the arguments for
        the function.       

        The type of all arguments and of the function return value should get specified using 
        standard Python type annotations. These types will get converted to json schema types.

        Each argument and the return value must be documented in the docstring. 

        If the type of a parameter is specified in the docstring, that type will get used
        instead of the type annotation specified in the function signature. 
        If the type of a parameter is specified in the docstring as a json schema type
        starting and ending with a brace, that schema is directly used. 

        See https://platform.openai.com/docs/guides/function-calling

        Args:
            functions: a function or list of functions. 

        Returns:
            A list of tool dictionaries, each dictionary describing a tool.
        """
        if not isinstance(functions, list):
            functions = [functions]
        tools = []
        for func in functions:
            tools.append(function2schema(func))
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

    def supports_file_upload(self, llmalias: str) -> bool:
        """
        Check if the model supports file upload

        NOTE: LiteLLM itself does not seem to support this, so we set this false by default and add specific
        models or providers where we know it works.
        """
        return False

    def count_tokens(self, llmalias: Union[str, List[Dict[str, any]]], messages: List[Dict[str, any]]) -> int:
        """
        Count the number of tokens in the given messages. If messages is a string, convert it to a
        single user message first.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return token_counter(model=self.llms[llmalias]["llm"], messages=messages)

    def query(
            self,
            llmalias: str,
            messages: List[Dict[str, str]],
            tools: Optional[List[Dict]] = None,
            return_cost: bool = False,
            return_response: bool = False,
            debug=False,
            litellm_debug=None,
            stream=True,
            recursive_call_info: Optional[Dict[str, any]] = None,  
            **kwargs,
    ) -> Dict[str, any]:
        """
        Query the specified LLM with the given messages.

        Args:
            llmalias: the alias/name of the LLM to query
            messages: a list of message dictionaries with role and content keys
            tools: a list of tool dictionaries, each dictionary describing a tool. 
                See https://docs.litellm.ai/docs/completion/function_call for the format. 
                However, this can be created using the `make_tooling` function.
            return_cost: whether or not LLM invocation costs should get returned
            return_response: whether or not the complete reponse should get returned
            debug: if True, emits debug messages to aid development and debugging
            litellm_debug: if True, litellm debug logging is enabled, if False, disabled, if None, use debug setting
            stream: if True, the returned object containst the stream that can be iterated over. Streaming
                may not work for all models.
            recursive_call_info: internal use only
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
        if recursive_call_info is None:
            recursive_call_info = {}            
        if llm.get("api_key"):
            completion_kwargs["api_key"] = llm["api_key"]
        elif llm.get("api_key_env"):
            completion_kwargs["api_key"] = os.getenv(llm["api_key_env"])
        if llm.get("api_url"):
            completion_kwargs["api_base"] = llm["api_url"]
        if tools is not None:
            # add tooling-related arguments to completion_kwargs
            completion_kwargs["tools"] = tools
            if not self.supports_function_calling(llmalias):
                # see https://docs.litellm.ai/docs/completion/function_call#function-calling-for-models-wout-function-calling-support
                litellm.add_function_to_prompt = True
            else:
                if "tool_choice"  not in completion_kwargs:
                    # this is the default, but lets be explicit
                    completion_kwargs["tool_choice"] = "auto"  
                # Not known/supported by litellm, apparently
                # if "parallel_tool_choice" not in completion_kwargs:
                #     completion_kwargs["parallel_tool_choice"] = True 
            fmap = toolnames2funcs(tools)
        else:
            fmap = {}
        if stream:
            # TODO: check if model supports streaming
            # if streaming is enabled, we always return the original response
            return_response = True
            completion_kwargs["stream"] = True
        ret = {}
        # before adding the kwargs, save the recursive_call_info and remove it from kwargs
        if debug:
            print(f"DEBUG: Received recursive call info: {recursive_call_info}")
        if kwargs:
            completion_kwargs.update(dict_except(kwargs,  KNOWN_LLM_CONFIG_FIELDS, ignore_underscored=True))
        if debug:
            print(f"DEBUG: Calling completion with kwargs {completion_kwargs}")
        # if we have min_delay set, we look at the _last_request_time for the LLM and caclulate the time
        # to wait until we can send the next request and then just wait
        min_delay = llm.get("min_delay", kwargs.get("min_delay", 0.0))
        if min_delay > 0:
            elapsed = time.time() - llm["_last_request_time"]
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)
        llm["_last_request_time"] = time.time()
        if "min_delay" in completion_kwargs:
            raise ValueError("Error: min_delay should not be passed as a keyword argument")
        try:
            # if we have been called recursively and the recursive_call_info has a start time, 
            # use that as the start time
            if recursive_call_info.get("start") is not None:
                start = recursive_call_info["start"]
            else:
                start = time.time()
                recursive_call_info["start"] = start
            response = litellm.completion(
                model=llm["llm"],
                messages=messages,
                **completion_kwargs)
            if stream:
                def chunk_generator(model_generator, retobj):
                    try:
                        for chunk in model_generator:
                            choice0 = chunk["choices"][0]
                            if choice0.finish_reason == "stop":
                                break
                            content = choice0["delta"].get("content", "")
                            yield dict(error="", answer=content, ok=True)
                    except Exception as e:
                        yield dict(error=str(e), answer="", ok=False)
                    finally:
                        # TODO: add cost and elapsed time information into retobj
                        # litellm does not support cost on streaming responses
                        # response.__hidden_params["response_cost"] is 0.0
                        ret["cost"] = None
                        ret["elapsed_time"] = time.time() - start
                        pass
                if return_response:
                    ret["response"] = response
                ret["chunks"] = chunk_generator(response, ret)
                ret["ok"] = True
                ret["error"] = ""
                return ret
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
                # TODO: replace with response._hidden_params["response_cost"] ? 
                #     but what if cost not supported for the model?
                
                try:
                    ret["cost"] = completion_cost(
                        completion_response=response,
                        model=llm["llm"],
                        messages=messages,
                    )
                    if debug:
                        print(f"DEBUG: cost for this call {ret['cost']}")
                except Exception as e:
                    logger.debug(f"Error in completion_cost for model {llm['llm']}: {e}")
                    ret["cost"] = 0.0
                llm["_cost"] += ret["cost"]
                usage = response['usage']
                logger.debug(f"Usage: {usage}")
                ret["n_completion_tokens"] = usage.completion_tokens
                ret["n_prompt_tokens"] = usage.prompt_tokens
                ret["n_total_tokens"] = usage.total_tokens
                # add the cost and tokens from the recursive call info, if available
                if recursive_call_info.get("cost") is not None:
                    ret["cost"] += recursive_call_info["cost"]
                    if debug:
                        print(f"DEBUG: cost for this and previous calls {ret['cost']}")
                if recursive_call_info.get("n_completion_tokens") is not None:
                    ret["n_completion_tokens"] += recursive_call_info["n_completion_tokens"]
                if recursive_call_info.get("n_prompt_tokens") is not None:
                    ret["n_prompt_tokens"] += recursive_call_info["n_prompt_tokens"]
                if recursive_call_info.get("n_total_tokens") is not None:
                    ret["n_total_tokens"] += recursive_call_info["n_total_tokens"]
                recursive_call_info["cost"] = ret["cost"]
                recursive_call_info["n_completion_tokens"] = ret["n_completion_tokens"]
                recursive_call_info["n_prompt_tokens"] = ret["n_prompt_tokens"]
                recursive_call_info["n_total_tokens"] = ret["n_total_tokens"]
                    
            response_message = response['choices'][0]['message']
            # Does not seem to work see https://github.com/BerriAI/litellm/issues/389
            # ret["response_ms"] = response["response_ms"]
            ret["finish_reason"] = response['choices'][0].get('finish_reason', "UNKNOWN")
            ret["answer"] = response_message['content']
            ret["error"] = ""
            ret["ok"] = True
            # TODO: if feasable handle all tool calling here or in a separate method which does
            #   all the tool calling steps (up to a specified maximum).
            if debug:
                print(f"DEBUG: checking for tool_calls: {response_message}, have tools: {tools is not None}")
            if tools is not None:
                # TODO: if streaming is enabled we need to gather the complete response before
                #   we can process the tool calls
                if hasattr(response_message, "tool_calls") and response_message.tool_calls is not None:
                    tool_calls = response_message.tool_calls
                else:
                    tool_calls = []
                if stream:
                    raise ValueError("Error: streaming is not supported for tool calls yet")
                if debug:
                    print(f"DEBUG: got {len(tool_calls)} tool calls:")
                    for tool_call in tool_calls:
                        print(f"DEBUG: {tool_call}")
                if len(tool_calls) > 0:   # not an empty list
                    if debug:
                        print(f"DEBUG: appending response message: {response_message}")
                    messages.append(response_message)
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        if debug:
                            print(f"DEBUG: tool call {function_name}")
                        fun2call = fmap.get(function_name)
                        if fun2call is None:
                            ret["error"] = f"Unknown tooling function name: {function_name}"
                            ret["answer"] = ""
                            ret["ok"] = False
                            return ret
                        function_args = json.loads(tool_call.function.arguments)
                        try:
                            if debug:
                                print(f"DEBUG: calling {function_name} with args {function_args}")
                            function_response = fun2call(**function_args)
                            if debug:
                                print(f"DEBUG: got response {function_response}")
                        except Exception as e:
                            tb = traceback.extract_tb(e.__traceback__)
                            filename, lineno, funcname, text = tb[-1]
                            if debug:
                                print(f"DEBUG: function call got error {e}")
                            ret["error"] = f"Error executing tool function {function_name}: {str(e)} in {filename}:{lineno} {funcname}"
                            if debug:
                                logger.error(f"Returning error: {e}")
                            ret["answer"] = ""
                            ret["ok"] = False
                            return ret
                        messages.append(
                            dict(
                                tool_call_id=tool_call.id, 
                                role="tool", name=function_name, 
                                content=json.dumps(function_response)))
                    # recursively call query
                    if debug:
                        print(f"DEBUG: recursively calling query with messages:")
                        for idx, msg in enumerate(messages):
                            print(f"DEBUG: Message {idx}: {msg}")
                        print(f"DEBUG: recursively_call_info is {recursive_call_info}")                    
                    return self.query(
                        llmalias, 
                        messages, 
                        tools=tools, 
                        return_cost=return_cost, 
                        return_response=return_response, 
                        debug=debug, 
                        litellm_debug=litellm_debug, 
                        recursive_call_info=recursive_call_info,
                        **kwargs)
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

    def supports_file_upload(self) -> bool:
        return self.llmsobject.supports_file_upload(self.config["alias"])

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        return self.llmsobject.count_tokens(self.config["alias"], messages)
