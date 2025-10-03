"""Module to simplify inspection of current loguru settings"""

from loguru import logger

def handlers() -> dict:
    return logger._core.handlers

def is_enabled(module: str) -> bool:
    al = logger._core.activation_list
    module = module + "."
    return next((x for x in al if x[0] == module), None)

