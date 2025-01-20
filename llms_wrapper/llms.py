"""
Module related to using LLMs.
"""
import os

import litellm
from loguru import logger
from typing import Optional, Dict, List, Union, Tuple
from copy import deepcopy

from litellm import completion, completion_cost
from litellm.utils import get_model_info
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
        self.llms = {}
        for llm in self.config["llms"]:
            alias = llm["alias"]
            if alias in self.llms:
                raise ValueError(f"Error: Duplicate LLM alis {alias} in configuration")
            self.llms[alias] = llm
            self.llms[alias]["cost"] = 0

    def list_models(self) -> List[Dict]:
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

    def __getitem__(self, item: str) -> Dict:
        """
        Get the LLM configuration object with the given alias.
        """
        return self.llms[item]

    def cost(self, llmalias: Union[str, List[str], None] = None):
        """
        Return the cost accumulated so far for the given llm alias given list of llm aliases
        or all llms if llmalias is None. Costs are only accumulated for invocations of
        the query method with return_cost=True.
        """
        if llmalias is None:
            return sum([llm["cost"] for llm in self.llms.values()])
        if isinstance(llmalias, str):
            return self.llms[llmalias]["cost"]
        return sum([self.llms[alias]["cost"] for alias in llmalias])

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

    def make_messages(
            self, query: Optional[str] = None, prompt: Optional[Dict[str, str]] = None,
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

    def query(
            self,
            llmalias: str,
            messages: List[Dict[str, str]],
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
                "api_key_env", "user_env", "password_env", "cost"])
        error = None
        if llm.get("api_key"):
            completion_kwargs["api_key"] = llm["api_key"]
        elif llm.get("api_key_env"):
            completion_kwargs["api_key"] = os.getenv(llm["api_key_env"])
        if llm.get("api_url"):
            completion_kwargs["api_base"] = llm["api_url"]

        ret = {}
        if debug:
            logger.debug(f"Calling completion with {completion_kwargs}")
        try:
            response = completion(
                model=llm["llm"],
                messages=messages,
                **completion_kwargs)
            logger.debug(f"Full Response: {response}")
            if return_response:
                ret["response"] = response
                ret["kwargs"] = completion_kwargs
            if return_cost:
                ret["cost"] = completion_cost(
                    completion_response=response,
                    model=llm["llm"],
                    messages=messages,
                )
                llm["cost"] += ret["cost"]
                usage = response['usage']
                logger.debug(f"Usage: {usage}")
                ret["n_completion_tokens"] = usage.completion_tokens
                ret["n_prompt_tokens"] = usage.prompt_tokens
                ret["n_total_tokens"] = usage.total_tokens
            ret["answer"] = response['choices'][0]['message']['content']
            ret["error"] = ""
            ret["ok"] = True
        except Exception as e:
            ret["error"] = str(e)
            if debug:
                logger.error(f"Returning error: {e}")
            ret["answer"] = ""
            ret["ok"] = False
        return ret

