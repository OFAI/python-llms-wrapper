# Quickstart/Examples

### 1) Install the package into your environment (see [[Installation]])

### 2) Create a config file for the LLM or LLMs you want to use:

e.g. `myconf.hjson`

```
{
  llms: [
    { llm: "openai/gpt-4o"
      api_key_env: "OPENAI_KEY1"
      temperature: 0
      alias: "openai1"
    }
    { llm: "gemini/gemini-1.5-flash", temperature: 1, alias: "geminiflash1" }
    { llm: "gemini/gemini-1.5-flash", temperature: 0, alias: "geminiflash2" }
]
  providers: {
    gemini: {
      api_key_env: "GEMINI_KEY1"
    }
  }
}
```

### 3) In your code: 

* import the class `LLMS` from `llms_wrapper.llms`
* read the configuration file 
* instantiate the class with the configuration object
* create a messages object, optionally using the `make_messages` method
* invoke any of the configured LLMs using the `query` method by specifying the alias of the LLM and passing on the prompt messages. 
  This can also optionally return the cost of the invocation and/or the full response object. 
* in order to invoke all LLMs configured, you can iterate over all aliases using the `list_alias` method

```
from llms_wrapper.llms import LLMS
from llms_wrapper.config import read_config_file

# [....]
config = read_config_file("myconf.hjson")
llms = LLMS(config)

messages = llms.make_messages(query="What is a monoid?")
response = llms.query("geminiflash", messages, return_cost=True)
if response["ok"]:
    print("Got the answer:", response["answer"])
    print("Cost:", response["cost"])
else:
    print("Got an error:", response["error"])
```
