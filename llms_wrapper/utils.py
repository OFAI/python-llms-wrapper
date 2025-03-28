"""
Module for various utility functions.
"""
import json


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