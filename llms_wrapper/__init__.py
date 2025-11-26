from loguru import logger
from llms_wrapper.version import __version__
logger.disable("llms_wrapper")

# prevent pdoc3 from loading llms_wrapper_webchat
__pdoc__ = {}
__pdoc__["llms_wrapper_webchat"] = False
