# Configuration

For every LLM to be used in a project, there should be configuration data that is necessary for using the LLM.
Configuration data is expected as a dictionary which contains the key `llms` associated with a list of per-LLM configurations
and optionally the key `providers` associated with a dictionary mapping provider names to settings common to all LLMs of that 
provider. 

Here is an example configuration showing a subset of the settings:
```
{
   "llms" : [
      {
         "api_key_env" : "OPENAI_KEY1",
         "llm" : "openai/gpt-4o",
         "temperature" : 0
      },
      {
         "llm" : "gemini/gemini-1.5-flash",
         "temperature" : 1,
         "alias": "gemini1"
      },
      {
         "llm" : "gemini/gemini-1.5-flash",
         "temperature" : 0,
         "alias": "gemini2"
      }
   ],
   "providers" : {
      "gemini" : {
         "api_key_env" : "GEMINI_KEY1"
      }
   }
}
```

Each llm is identified by the provider name, e.g. "openai" or "gemini" followed by a slash, followed by the model name or id. Provider 
names must match the provider names known in the [litellm](https://docs.litellm.ai/docs/providers) package. 

Parameters specified for each of the providers in the `providers`  section apply to every llm in the `llms`  section unless the same 
parameter is also specified for the llm, in which case that value takes precedence.

The following parameters are known and supported in the `llms` and/or `providers` sections:

* `llm` (`llms` section only): specifies a specific model using the format `providername/modelid`. 
* `api_key`: the literal API key to use
* `api_key_env`:  the environment variable which contains the API key
* `api_url`: the base URL to use for the model, e.g. for an ollama server. The URL may contain placeholders which will get replaced with
   the model name (`${model}`), or the user and password for basic authentication (`${user}`, `${password}`), e.g.
   `http://${user}:${password}@localhost:11434`
* `user`, `password`: the user and password to use for basic authentication, this requires `api_url` to also be specified with the 
   corresponding placeholders
* `alias` (`llms` section only): an alias name for the model which will have to be used in the API. If no `alias` is specified, the name
   specified for `llm` is used. 
* `num_retries`: if present, can specify the number of retries to perform if an error occurs before giving up
* `timeout`: if present, raise timeout error after that many seconds

All other settings are passed as is to the model invocation function. Different providers or APIs may support different parameters, but 
most will support `temperature`, `max_tokens` and `top_p` 

IMPORTANT: the raw configuration as shown in the example above needs to get processed by the function `llms_wrapper.config.update_llm_config` 
in order to perform all the necessary substitutions!

#### Other top-leve config entries

Currently the following other top level configuration fields in addition to `llms` and `providers` are recognized:

* `use_phoenix`: if present, should be the URL of a local phoenix endpoint or a list containing the endpoint URL and the project name

## Configuration files

The configuration as described above can be created programmatically or read in from some config file. 
The function `llms_wrapper.config.read_config_file`  allows reading the configuration from files in any of the following 
formats: json, hsjson, yaml, toml. By default, reading the configuration that way will also perform the necessary substitutions by
automatically invoking `llms_wrapper.config.update_llm_config` 

