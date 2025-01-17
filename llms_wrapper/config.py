"""
Module for reading config files.

The config file can be in
one of the following formats: json, hjson, yaml, toml. This module only cares about the top-level fields
"llms" and "providers": all other fields are ignored.

"""
from loguru import logger
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
        if not re.match(r"^[a-zA-Z0-9]+/\S+$", llm):
            raise ValueError(f"Error: 'llm' field must be in the format 'provider/model' in line: {llm}")
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
    for i, llm in enumerate(config["llms"]):
        if isinstance(llm, str):
            provider, model = llm.split(":")
            if provider in config.get("providers", {}):
                provider_config = config["providers"][provider]
                llm = {
                    "llm": llm,
                    "api_key": provider_config.get("api_key", os.getenv(f"{provider.upper()}_API_KEY"))
                }
                # copy over user, password, user_env, and password_env if they exist
                if "user" in provider_config:
                    llm["user"] = provider_config["user"]
                if "password" in provider_config:
                    llm["password"] = provider_config["password"]
                if "user_env" in provider_config:
                    llm["user"] = os.getenv(provider_config["user_env"])
                if "password_env" in provider_config:
                    llm["password"] = os.getenv(provider_config["password_env"])
            else:
                llm = {
                    "llm": llm,
                    "api_key": os.getenv(f"{provider.upper()}_API_KEY")
                }
        else:
            provider, model = llm["llm"].split("/", 1)
            if "api_key" not in llm:
                if "api_key_env" in llm:
                    llm["api_key"] = os.getenv(llm["api_key_env"])
                else:
                    if provider in config.get("providers", {}):
                        provider_config = config["providers"][provider]
                        llm["api_key"] = provider_config.get("api_key", os.getenv(f"{provider.upper()}_API_KEY"))
                    else:
                        llm["api_key"] = os.getenv(f"{provider.upper()}_API_KEY")
        if not llm.get("password") and "password_env" in llm:
            llm["password"] = os.getenv(llm["password_env"])
        if not llm.get("user") and "user_env" in llm:
            llm["user"] = os.getenv(llm["user_env"])
        if not llm.get("api_key") and "api_key_env" in llm:
            llm["api_key"] = os.getenv(llm["api_key_env"])
        config["llms"][i] = llm
        if "api_url" in llm:
            if "api_key" in llm and llm["api_key"]:
                llm["api_url"] = llm["api_url"].replace("${api_key}", llm["api_key"])
            if "user" in llm:
                llm["api_url"] = llm["api_url"].replace("${user}", llm["user"])
            if "password" in llm:
                llm["api_url"] = llm["api_url"].replace("${password}", llm["password"])
            if "model" in llm:
                llm["api_url"] = llm["api_url"].replace("${model}", llm["model"])
        # if there is no alias defined, set the alias to the model name
        if not "alias" in llm:
            llm["alias"] = llm["llm"]
    # make sure all the aliases are unique
    aliases = set()
    for llm in config["llms"]:
        if llm["alias"] in aliases:
            raise ValueError(f"Error: Duplicate alias {llm['alias']} in LLM list")
        aliases.add(llm["alias"])
    return config