#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module to perform a simple test if one or more LLMs are working
"""
from loguru import logger
import sys
import argparse
import re
from llms_wrapper.llms import LLMS
from llms_wrapper.config import read_config_file, update_llm_config
from llms_wrapper.utils import pp_config

DEFAULT_PROMPT = """What is the first name of Einstein who developed the theory of relativity? 
Only give the name and no additional text?"""

DEFAULT_ANSWER = "Albert"


def get_args():
    """
    Get the command line arguments
    """
    parser = argparse.ArgumentParser(description='Test llms')
    parser.add_argument('--llms', '-l', nargs="*", type=str, default=[], help='LLMs to use for the queries (or use config)', required=False)
    parser.add_argument("--prompt", "-p", type=str, help="Prompt text to use (or use default prompt)", required=False)
    parser.add_argument("--answer", "-a", type=str, help="Expected answer (or use default answer)", required=False)
    parser.add_argument("--config", "-c", type=str, help="Config file with the LLM and other info for an experiment, json, jsonl, yaml", required=False)
    parser.add_argument('--role', '-r', choices=["user", "system", "assistant"], default="user", help='Role to use for the prompt', required=False)
    parser.add_argument("--dry-run", "-n", action="store_true", help="Dry run, do not actually run the queries", required=False)
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode", required=False)
    parser.add_argument("--show_response", action="store_true", help="Show thefull  response from the LLM", required=False)
    parser.add_argument("--show_cost", action="store_true", help="Show token counts and cost", required=False)
    parser.add_argument("--logfile", "-f", type=str, help="Log file", required=False)
    args = parser.parse_args()
    for llm in args.llms:
        if not re.match(r"^[a-zA-Z0-9_\-./]+:[a-zA-Z0-9_\-./]+$", llm):
            raise Exception(f"Error: 'llms' field must be in the format 'provider:model' in: {llm}")
    # convert the argparse object to a dictionary
    argsconfig = {}
    argsconfig.update(vars(args))

    # if a config file is specified, read the config file using our config reading function and update the arguments.
    # The config data may contain:
    # - input: used only if not specified in the command line arguments
    # - output: used only if not specified in the command line arguments
    # - llm: added to the ones specified in the command line arguments
    # - prompt: used to add config info to the llms specified in the command line arguments
    if args.config:
        config = read_config_file(args.config, update=False)
        # merge the args into the config, giving precedence to the args, except for the LLM list, which is merged
        # by adding the args to the config
        oldllm = config.get("llms", [])
        config.update(argsconfig)
        # add the llm from the args to the llm from the config, but only if the llm is not already in the config
        mentionedllm = [llm if isinstance(llm, str) else llm["llm"] for llm in config["llms"]]
        for llm in args.llms:
            if llm not in mentionedllm:
                oldllm.append(llm)
        config["llms"] = oldllm
    else:
        config = argsconfig
    update_llm_config(config)
    config["answer"] = args.answer
    # make sure we got at least one llm
    if not config["llms"]:
        raise Exception("Error: No LLMs specified")
    return config


def equal_response(response, answer):
    """
    Check if the response is equal to the answer, disregarding any newlines and multiple spaces and any
    leading or trailing spaces.
    """
    return response.replace("\n", " ").replace("  ", " ").strip() == answer

def run(config: dict):
    prompt = {}
    prompt[config['role']] = config['prompt'] if config['prompt'] else DEFAULT_PROMPT
    answer = config['answer'] if config['answer'] else DEFAULT_ANSWER
    n = 0
    n_ok = 0
    n_nok = 0
    log = []
    llms = LLMS(config)
    messages = llms.make_messages(prompt=prompt)
    for alias in llms.list_aliases():
        llmname = alias
        llm = llms.get(alias)
        n += 1
        if config['dry_run']:
            logger.info(f"Would query LLM {llmname} with prompt {prompt}")
            n_ok += 1
        else:
            if config['debug']:
                apikey = llm['api_key'] if isinstance(llm, dict) else "NONE"
                logger.debug(f"Querying LLM {llmname} apikey {apikey}  with prompt {prompt}")
            ret = llms.query(
                llmname,
                messages=messages,
                debug=config['debug'],
                return_response=config["show_response"],
                return_cost=config["show_cost"])
            response = ret.get("answer", "")
            error = ret.get("error", "")
            ret_response = ret.get("response", "")
            ret_cost = ret.get("cost", "")
            ret_completion_tokens = ret.get("n_completion_tokens", 0)
            ret_prompt_tokens = ret.get("n_prompt_tokens", 0)
            ret_total_tokens = ret.get("n_total_tokens", 0)
            if config["show_response"]:
                logger.info(f"Response/error from {llmname}: {response}/{error}")
                logger.info(f"Detailed response: {ret_response}")
            if config["show_cost"]:
                logger.info(f"Cost for {llmname}: {ret_cost}")
                logger.info(f"Completion tokens: {ret_completion_tokens}, Prompt tokens: {ret_prompt_tokens}, Total tokens: {ret_total_tokens}")
            if error:
                n_nok += 1
                logger.error(f"Error from {llmname}: {error}")
                log.append(f"{llmname} Error: {error}")
            elif not equal_response(response, answer):
                logger.error(f"Error: Unexpected response from {llmname}: {response}, expected: {answer}")
                log.append(f"{llmname} Unexpected response:  {response}, expected: {answer}")
            else:
                n_ok += 1
                if config['debug']:
                    logger.info(f"OK Response from {llmname}: {response}")
                log.append(f"{llmname} OK")

    logger.info("Summary:")
    for l in log:
        logger.info(l)
    logger.info(f"OK: {n_ok}, NOK: {n_nok}")


def main():
    args = get_args()
    if args["logfile"]:
        logger.add(args["logfile"])
    if args["debug"]:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        ppargs = pp_config(args)
        logger.debug(f"Effective arguments: {ppargs}")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    run(args)


if __name__ == '__main__':
    main()
