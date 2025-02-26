"""
Module which implements an approach to find the available models for a provider, trying to use the
underlying API if possible, and falling back to direct HTTP requests if necessary.
"""
import litellm


# litellm model list attributes:
# [n for n in litellm.__dict__.keys() if n.endswith("models")]

def model_list(provider, api_key=None):
    """
    Return a list of models available for the given provider. If the provider is not recognized,
    or the underlying API is not available, return None.

    api_key should not be necessary but parameter is present if an alternate method to get the list
    requires it
    """
    if provider == "openai":
        try:
            from openai import OpenAI
            client = OpenAI()
            return client.models.list()
        except ImportError:
            return None
    elif provider == "gemini":
        try:
            return litellm.gemini_models
        except ImportError:
            return None
    elif provider == "anthropic":
        try:
            return litellm.anthropic_models
        except ImportError:
            return None
    elif provider == "mistral":
        try:
            return litellm.mistral_chat_models
        except ImportError:
            return None
    elif provider == "xai":
        try:
            return litellm.xai_models
        except ImportError:
            return None
    elif provider == "groq":
        try:
            return litellm.groq_models
        except ImportError:
            return None
    elif provider == "palm":
        try:
            return litellm.palm_models
        except ImportError:
            return None
    elif provider == "perplexity":
        try:
            return litellm.perplexity_models
        except ImportError:
            return None
    if provider is None or provider == "":
        # for now use all the models from all providers as known to litellm
        ret = []
        alist = [n for n in litellm.__dict__.keys() if n.endswith("models")]
        for a in alist:
            if a in [
                "caching_with_models", "add_known_models",
                "ahealth_check_wildcard_models", "all_embedding_models"
            ]:
                continue
            tmp = getattr(litellm, a)
            if not isinstance(tmp, list):
                continue
            tmp = [f"{a}/{m}" for m in tmp]
            ret += tmp
        return ret
    else:
        return None