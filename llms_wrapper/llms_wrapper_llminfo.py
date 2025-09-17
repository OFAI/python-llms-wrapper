#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module for the llms_wrapper_llminfo command to show known properties of one or more LLMs defined in a config file.
"""
import sys
import argparse
import re
from loguru import logger
from litellm import get_model_info
from llms_wrapper.log import configure_logging
from llms_wrapper.llms import LLMS
from llms_wrapper.config import read_config_file, update_llm_config
from llms_wrapper.utils import pp_config, dict_except
from llms_wrapper.version import __version__


def get_args() -> dict:
    """
    Get the command line arguments
    """
    parser = argparse.ArgumentParser(description='Show known LLM properties')
    parser.add_argument('--llms', '-l', nargs="+", type=str, default=[], help='LLM names, or aliases from the config file for which to get info (all)', required=False)
    parser.add_argument('--config', '-c', type=str, help='Config file with LLM definitions', required=False)
    parser.add_argument('--full', '-f', action="store_true", help='Include info for which an API key is required', required=False)
    parser.add_argument('--short', '-s', action="store_true", help='Only show one-line info per model, ignores --full', required=False)
    parser.add_argument("--debug", action="store_true", help="Debug mode", required=False)
    args = parser.parse_args()
    loglevel1 = "INFO"
    if args.debug:
        loglevel1 = "DEBUG"
    configure_logging(level=loglevel1)
    for llm in args.llms:
        if not re.match(r"^[a-zA-Z0-9_\-./]+/.+$", llm):
            raise Exception(f"Error: 'llms' field must be in the format 'provider/model' in: {llm}")
    # convert the argparse object to a dictionary
    argsconfig = {}
    argsconfig.update(vars(args))
    return argsconfig


def run(args: dict):
    if args["config"]:
        config = read_config_file(args["config"], update=True)
        # get all the aliases of the llms
        aliases = [llm["alias"] for llm in config["llms"]]
        # if no llms are specified as arguments, use all the aliases from the config
        use_llmnames = aliases if len(args["llms"]) == 0 else args["llms"]
    else:
        use_llmnames = args["llms"]
        # create a fake config which has one entry
        config = {"llms": [dict(llm=llm) for llm in use_llmnames]}
    if len(use_llmnames) == 0:
        print("No LLMs specified with --llm or in a config file specified with --config")
        sys.exit(1)
    llms = LLMS(config)
    for llmalias in use_llmnames:
        llmname = llms[llmalias]["llm"]
        model_known = False
        try:
            model_info = get_model_info(llmname)
            model_known = True
        except Exception as e:
            model_known = False
        if not model_known:
            print(f"LLM: {llmname} (unknown)")
            continue
        info = get_model_info(llmname)
        if args["short"]:
            print(f"{llmalias}: {llmname} max in/out: {llms.max_input_tokens(llmname)}/{llms.max_output_tokens(llmname)} cost in/out: {llms.cost_per_token(llmname)[0]*1000000:.2f}/{llms.cost_per_token(llmname)[1]*1000000:.2f}")
            continue
        print(f"LLM: {llmname}")
        print(f"Max number of prompt tokens:                  {llms.max_input_tokens(llmname)}")
        print(f"Max number of output tokens:                  {llms.max_output_tokens(llmname)}")
        print(f"Cost per 1M prompt tokens (estimate):         {llms.cost_per_token(llmname)[0]*1000000:.2f}")
        print(f"Cost per 1M output tokens (estimate):         {llms.cost_per_token(llmname)[1]*1000000:.2f}")
        print(f"Supports system messages:                     {info['supports_system_messages']}")
        print(f"Supports function calling:                    {info['supports_system_messages']}")
        print(f"Supports tool choice:                         {info['supports_tool_choice']}")
        for p in ["temperature", "top_p", "max_tokens", "max_completion_tokens", "tools", "tool_choice", "functions", "response_format", "n", "stop", "logprobs", "frequency_penalty", "presence_penalty"]:
            spc = " "*(26-len(p))
            print(f"Supports parameter {p}:{spc}{p in info['supported_openai_params']}")
        #print(f"Info:\n{info}")


def main():
    args = get_args()
    run(args)


if __name__ == '__main__':
    logger.enable("llms_wrapper")
    main()
