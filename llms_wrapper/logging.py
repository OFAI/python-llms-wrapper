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
from loguru import logger

DEFAULT_LOGGING_LEVEL = "INFO"
DEFAULT_LOGGING_FORMAT = "{time} {level} {module}: {message}"

def configure_logging(level=None, logfile=None, format=None, enable=True):
    """
    Configure loguru logging sinks. This removes the default sink and adds one for stderr and, if a logfile
    is specified, one for the logfile, both for the specified level. The format of the log messages can be
    specified with the format parameter or the default format is used.
    """
    logger.remove()
    if level is None:
        level = DEFAULT_LOGGING_LEVEL
    if format is None:
        format = DEFAULT_LOGGING_FORMAT
    logger.add(sys.stderr, level=level, format=format)
    if logfile is not None:
        logger.add(logfile, level=level, format=format)
    sys.excepthook = handle_exception
    if enable:
        logger.enable("llms_wrapper")


# Define a custom exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.opt(exception=(exc_type, exc_value, exc_traceback)).error("Unhandled exception")




