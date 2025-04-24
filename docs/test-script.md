After installing the `llms_wrapper`  package the command `llms_wrapper_test` is available as a new command.

The command can be used to run a standard query against all LLMs configured and check the response (the default query asks the LLM 
for the first name of Einstein and checks for the name "Albert" to be returned). 

In order to get usage information run the command with the `--help` option:

```
usage: llms_wrapper_test [-h] [--llms [LLMS ...]] [--use [USE ...]] [--prompt PROMPT] [--answer ANSWER] [--config CONFIG]
                         [--role {user,system,assistant}] [--dry-run] [--debug] [--show_response] [--show_cost]
                         [--logfile LOGFILE]

Test llms

options:
  -h, --help            show this help message and exit
  --llms [LLMS ...], -l [LLMS ...]
                        LLMs to use for the queries (or use config)
  --use [USE ...], -u [USE ...]
                        Subset of LLMs to use (all)
  --prompt PROMPT, -p PROMPT
                        Prompt text to use (or use default prompt)
  --answer ANSWER, -a ANSWER
                        Expected answer (or use default answer)
  --config CONFIG, -c CONFIG
                        Config file with the LLM and other info for an experiment, json, jsonl, yaml
  --role {user,system,assistant}, -r {user,system,assistant}
                        Role to use for the prompt
  --dry-run, -n         Dry run, do not actually run the queries
  --debug, -d           Debug mode
  --show_response       Show thefull response from the LLM
  --show_cost           Show token counts and cost
  --logfile LOGFILE, -f LOGFILE
                        Log file
```

In order to run a quick test without creating a config file, it is possible to 
just specify the name of the model(s) with the `-llms` parameter and set the corresponding API 
key via the standard environment variable, e.g.

```
export GROQ_API_KEY=${MY_SPECIAL_GROQ_API_KEY} llms_wrapper_test --llms  groq/llama3-70b-8192
```