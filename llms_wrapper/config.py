"""
Module for reading config files.

The config file can be in
one of the following formats: json, hjson, yaml, toml. This module only cares about the top-level fields
"llms" and "providers": all other fields are ignored.

"""
import warnings

## For debugging the stacktrace of the weird litellm warning
## import traceback
## def custom_warning_format(message, category, filename, lineno, file=None, line=None):
##     return f"{filename}:{lineno}: {category.__name__}: {message}\n{''.join(traceback.format_stack())}"
## warnings.formatwarning = custom_warning_format

import os
import json
import yaml
import hjson
import tomllib
import re
from dotenv import load_dotenv

## Suppress the annoying litellm warning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from litellm import LITELLM_CHAT_PROVIDERS


def read_config_file(filepath: str, update: bool = True) -> dict:
    """
    Read a config file in one of these formats: json, hjson, yaml, toml. Return the dict with the configuration.
    This function already checks that the llm-related fields "llms" and "proviers" in the config file are valid.

    - llms: a list of strings or dicts with the LLM name and the
        LLM config to use. If the LLM is identified by a string it has to be in the format 'provider/model' according
        to the LightLLM naming scheme. 'provider' is the provider name used by the litellm backend but it always
        has to be present, even if it is optional in litellm. See https://docs.litellm.ai/docs/providers
        The LLM config is a dict with the following fields:
        - api_key: the API key to use for the LLM
        - api_key_env: the name of the environment variable to use for the API key. Ignored if api_key is specified.
        - api_url: the URL to use. In this URL, the placeholders ${model}, ${user}, ${password}, ${api_key}
            are replaced with the actual values.
        - user: the user name to use for basic authentication
        - password: the password to use for basic authentication
        - alias: a user friendly unique name for the LLM model (provider, model and settings=. The alias must be
          unique among all LLMs in the config file. If not specified, the provider+modelname is used as the alias.
        - OTHER FIELDS are passed to the LLM as is, however most providers just support the following additional
          fields: temperature, max_tokens, top_p
        If config settings are specified in both a provider config and an llm config for the provider, the
        settings in the llm config take precedence.
    - providers: a dict with with LLM provider names and a dict of config settings for the provider. The follogin
        fields are allowed in the provider config:
        - api_key: the API key to use for the LLM
        - api_key_env: the name of the environment variable to use for the API key
        - api_url: the URL to use. In this URL, the placeholders ${model}, ${user}, ${password}, ${api_key}
            are replaced with the actual values.
        - user: the user name to use for basic authentication
        - password: the password to use for basic authentication

    Note that config files without a "llms" field are allowed and will be treated as if the "llms" field is an empty list.
    The same is true for the "providers" field.

    Args:
        filepath: where to read the config file from
        update: if True, update the LLM information in the config dict for each LLM in the list

    Returns:
        A dict with the configuration
    """
    # read config file as json, yaml or toml, depending on file extension
    load_dotenv(override=True)
    if filepath.endswith(".json"):
        with open(filepath, 'r') as f:
            config = json.load(f)
    if filepath.endswith(".hjson"):
        with open(filepath, 'r') as f:
            config = hjson.load(f)
    elif filepath.endswith(".yaml"):
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
    elif filepath.endswith(".toml"):
        with open(filepath, 'r') as f:
            config = tomllib.load(f)
    else:
        raise ValueError(f"Unknown file extension for config file {filepath}")
    if not "llms" in config:
        config["llms"] = []
    else:
        if not isinstance(config["llms"], list):
            raise ValueError(f"Error: 'llms' field in config file {filepath} must be a list")
    if not "providers" in config:
        config["providers"] = {}
    else:
        if not isinstance(config["providers"], dict):
            raise ValueError(f"Error: 'providers' field in config file {filepath} must be a dict")
    for llm in config["llms"]:
        if not isinstance(llm, str) and not isinstance(llm, dict):
            raise ValueError(f"Error: LLM entry in config file {filepath} must be a string or a dict")
        if isinstance(llm, dict):
            if not 'llm' in llm:
                raise ValueError(f"Error: Missing 'llm' field in llm config")
            llm = llm["llm"]
        if not re.match(r"^[a-zA-Z0-9_-]+/.+$", llm):
            raise ValueError(f"Error: 'llm' field must be in the format 'provider/model' in line: {llm}")
        # add known additional configuration fields: these can get specified using a name like e.g. cost_per_prompt_token
        # but get stored in the config as _cost_per_prompt_token to avoid passing them to the LLM.
        # All other fields, i.e. fields with unknown names are passed to the LLM as is.
    for provider, provider_config in config['providers'].items():
        # provider name must be one of the supported providers by litellm
        if provider not in LITELLM_CHAT_PROVIDERS:
            raise ValueError(f"Error: Unknown provider {provider}, must be one of {LITELLM_CHAT_PROVIDERS}")
        # all the fields are optional, but at least one should be specified
        if (not 'api_key' in provider_config and
                not 'api_url' in provider_config and
                not 'user' in provider_config and
                not 'password' in provider_config and
                not 'api_key_env' in provider_config and
                not 'user_env' in provider_config and
                not 'password_env' in provider_config
        ):
            raise ValueError(f"Error: Missing config settings for provider {provider}")
    if update:
        update_llm_config(config)
    return config


def update_llm_config(config: dict):
    """
    Update the LLM information in the config dict for each LLM in the list.

    This will make sure the information provided in the providers section of the config file
    is transferred to the llms and that other substitutions in the configuration are carried out
    for all llms.

    If the LLM is a string, replace it
    by a dict with all the details. The details are taken from the corresponding provider definition in the config
    file, if it exists, otherwise just the API key is taken from the default environment variable.
    The api key is selected in the following way: if the LLM dict speicifies it, use it, otherwise, if the LLM
    dict specifies api_key_env, use the environment variable with that name, otherwise use the api_key setting from
    the corresponding provider definition in the config file, otherwise use the api_key_env setting from the
    corresponding provider definition in the config file, otherwise use the default environment variable.
    In addition, for each llm, update the api_url field by replacing the placeholders ${api_key}, "${user}",
    "${password}", and "${model}" with the actual values.

    Args:
        config: the configuration dict to update. Note: this is modified in place!

    Returns:
        the updated configuration dict
    """
    load_dotenv(override=True)
    for i, llm in enumerate(config["llms"]):
        if isinstance(llm, str):
            provider, model = llm.split("/")
            if provider in config.get("providers", {}):
                llm = {}
                llm.update(config["providers"][provider])
            else:
                llm = {
                    "llm": llm,
                }
        else:
            if "/" not in llm["llm"]:
                raise ValueError(f"Error: LLM entry in config file must be in the format 'provider/model'")
            provider, model = llm["llm"].split("/", 1)
            provider_config = config.get("providers", {}).get(provider, {})
            for key in provider_config:
                if key not in llm:
                    llm[key] = provider_config[key]
        if "api_key" not in llm and "api_key_env" not in llm and os.environ.get(f"{provider.upper()}_API_KEY"):
            llm["api_key_env"] = f"{provider.upper()}_API_KEY"
        config["llms"][i] = llm
        if "api_url" in llm:
            # get the user, password and api_key for substitution
            user = llm.get("user")
            if user is None and "user_env" in llm:
                user = os.environ.get(llm["user_env"])
            password = llm.get("password")
            if password is None and "password_env" in llm:
                password = os.environ.get(llm["password_env"])
            api_key = llm.get("api_key")
            if api_key is None and "api_key_env" in llm:
                api_key = os.environ.get(llm["api_key_env"])
            if api_key is not None:
                llm["api_url"] = llm["api_url"].replace("${api_key}", api_key)
            if user is not None:
                llm["api_url"] = llm["api_url"].replace("${user}", user)
            if password is not None:
                llm["api_url"] = llm["api_url"].replace("${password}", password)
            llm["api_url"] = llm["api_url"].replace("${model}", model)
        # if there is no alias defined, set the alias to the model name
        if "alias" not in llm:
            llm["alias"] = llm["llm"]
    # make sure all the aliases are unique
    aliases = set()
    for llm in config["llms"]:
        if llm["alias"] in aliases:
            raise ValueError(f"Error: Duplicate alias {llm['alias']} in LLM list")
        aliases.add(llm["alias"])
    return config
