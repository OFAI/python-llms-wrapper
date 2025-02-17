# -*- coding: utf-8
"""
Module to handle logging. This module is used to set up the logging for the entire package. 
NOTE: by default, loguru logging is disabled for the package (see __init__.py). To enable logging,
a client of this library must setup its own logging configuration and enable logging for this package
using logger.enable("llms_wrapper"). 
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




