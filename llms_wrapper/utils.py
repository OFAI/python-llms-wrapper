"""
Module for various utility functions.
"""
import sys
import json
from typing import List, Dict
from loguru import logger

def pp_config(config):
    """
    Pretty print the config dict
    """
    return json.dumps(config, indent=4, sort_keys=True)

def dict_except(d: dict, keys, ignore_underscored=True):
    """
    Return a copy of the dict d, except for the keys in the list keys.
    """
    if isinstance(keys, str):
        keys = [keys]
    if ignore_underscored:
        keys = [k for k in d.keys() if k.startswith("_")] + keys
    return {k: v for k, v in d.items() if k not in keys}

# functions to load and convert prompts

def promptlist2dict(plist: List[dict]) -> Dict:
    new_prompts = {}
    for prompt in plist:
        if "pid" not in prompt:
            logger.error("Prompt does not contain a pid")
            sys.exit(1)
        if prompt["pid"] in new_prompts:
            logger.error(f"Duplicate pid {prompt['pid']} in prompts")
            sys.exit(1)
        new_prompts[prompt["pid"]] = prompt
    return new_prompts

def check_prompts(pdict: Dict):
    """
    Check the content of a prompts dictionary: each key is assumed to be a pid and the value is either a prompt represented
    as a dict (mapping role names to content strings/templates) or a list representing a standard message list
    (each dict in the list has the keys "role" and "content").

    If the value is a dict we make sure the pid is part of the dict and matches the key.
    """
    for pid, prompt in pdict.items():
        if isinstance(prompt, dict):
            if "pid" not in prompt:
                prompt["pid"] = pid
            if pid != prompt["pid"]:
                logger.error("Prompt pid does not match key in prompts dictionary")
                sys.exit(1)
        elif isinstance(prompt, list):
            pass # we assume this is a standard message list
        elif isinstance(prompt, str):
            # convert the string to a prompt dict with role user and the string being the content
            # TODO: this probably cannot be done while we are iterating over the dictionary!!!
            pdict[pid] = {"user": prompt}
        else:
            logger.error(f"Invalid prompt format, prompt for pid {pid} is neither a dict nor a list but {type(prompt)}")
            sys.exit(1)



def prompts_as_messages(prompts: Dict) -> Dict:
    """
    Convert a list of prompts represented by dicts mapping roles to content, or a dict of prompts where each
    value is a dict mapping roles to content to a dict of prompts where each value is a standard list of messages.
    If the prompts are already in message list format, they are returned unchanged.

    :param prompts:
    :return:
    """
    # check if the prompts are already in dict format, if not convert to dict. If the prompt dict does not
    # contain a pid, we use the string representation of the index as the pid
    if isinstance(prompts, list):
        # convert the list of prompts to a dict
        prompts = promptlist2dict(prompts)
    # now convert each prompt to message list format, if necessary
    ret = {}
    for pid, prompt in prompts.items():
        if isinstance(prompt, dict):
            # convert the prompt to a message list
            ret[pid] = [{"role": role, "content": content} for role, content in prompt.items() if role != "pid"]
        elif isinstance(prompt, list):
            # we assume this is already in message list format
            ret[pid] = prompt
        else:
            logger.error(f"Invalid prompt format, prompt for pid {pid} is neither a dict nor a list but {type(prompt)}")
            sys.exit(1)
    return ret


def load_prompts(fname: str, as_messages=False) -> Dict:
    """
    Load prompts from a json or hjson file and return a dictionary mapping pids to prompts.
    The following structures are supported:
    * a list of dictionaries, where each dictionary maps role names to content strings/templates and also has
      a pid key. If the pid key is missing the string representation of the index is used as the pid.
    * a dictionary mapping pids to dictionaries, where each dictionary maps role names to content strings/templates.
      The pid key is optional, if it is missing the key of the dictionary is used as the pid and added to the prompt
      dictionary.
    * a dictionary mapping pids to a standard list of messages, each message is a dict with the keys "role" and "content".

    If the as_messages flag is set to True, the prompts are returned as a dict where the key is the pid and each
    prompt is represented in standard message list format. Otherwise the prompts are returned as a dict where the key
    is the pid and each prompt is a dict mapping role names to content strings/templates.
    """
    if fname.endswith(".json"):
        import json
        with open(fname, "r") as f:
            prompts = json.load(f)
    elif fname.endswith(".hjson"):
        import hjson
        with open(fname, "r") as f:
            prompts = hjson.load(f)
    else:
        logger.error("Prompts file must be in json or hjson format")
        sys.exit(1)
    # if prompts is a dict and has a top-level key "prompts", we use the value of this key as the prompts
    if isinstance(prompts, dict) and "prompts" in prompts:
        prompts = prompts["prompts"]

    # the content should be either a list of prompts as a dictionary or a dictionary mapping pids to prompts
    # if it is a list, we convert it to a dictionary mapping pids to prompts and require every dict to contain a pid,
    # if it is a dict, we make sure every prompt gets the pid from the key or has already a pid that is identical to
    # the key
    if isinstance(prompts, list):
        prompts = promptlist2dict(prompts)
    elif isinstance(prompts, dict):
        check_prompts(prompts)
    else:
        logger.error("Prompts file must contain a list of prompts or a dictionary mapping pids to prompts")
        sys.exit(1)
    if as_messages:
        prompts = prompts_as_messages(prompts)
    return prompts
