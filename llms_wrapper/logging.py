# -*- coding: utf-8
"""
Module to handle logging. This module is used to set up the logging for the entire package. This uses the Python
logging module, but with a more user-friendly interface. Importing this module will provide a looger object
that is configured to log to stderr with a logging level INFO by default.

The format of the log messages is:  "{time} {level}: {message}"

The module provides a function set_logging_level to update the logging level of all handlers and also provides
the function add_logging_file to add a file handler to the specified file and the current logging level.
If set_logging_level is called after a file handler has been added, the logging level of the file handler is
changed too.
"""
import sys
import logging

# create a logger object
logger = logging.getLogger("llms_wrapper")
# set the logging level to INFO
logger.setLevel(logging.INFO)
# create a handler to log to stderr
handler = logging.StreamHandler(sys.stderr)
# create a formatter to format the log messages
formatter = logging.Formatter("{asctime} {levelname} {filename}/{funcName}:{lineno}: {message}", datefmt='%Y-%m-%d %H:%M:%S', style="{")
# set the formatter for the handler
handler.setFormatter(formatter)
# add the handler to the logger
logger.addHandler(handler)

# Define a custom exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Exception", exc_info=(exc_type, exc_value, exc_traceback))

# Set the custom exception handler as the default
sys.excepthook = handle_exception

def set_logging_level(level: int):
    """
    Set the logging level of all handlers to the specified level.
    :param level: the logging level to set
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def add_logging_file(file: str):
    """
    Add a file handler to the specified file and the current logging level.
    :param file: the file to log to
    """
    file_handler = logging.FileHandler(file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logger.level)
    logger.addHandler(file_handler)



