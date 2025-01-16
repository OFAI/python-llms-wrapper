"""
Module for various utility functions.
"""
import json


def pp_config(config):
    """
    Pretty print the config dict
    """
    return json.dumps(config, indent=4, sort_keys=True)

def dict_except(d, keys):
    """
    Return a copy of the dict d, except for the keys in the list keys.
    """
    return {k: v for k, v in d.items() if k not in keys}