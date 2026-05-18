# quick start TLDR; just fill in API key and run this:
```powershell
$env:STABILIZER_API_KEY = "YOUR_STABILIZER_API_KEY"
py -m stabilizer_python_sdk.run_me --new
```
# end quick start


# stabilizer-python-sdk

Standalone Python SDK and CLI for the Stabilizer API at `https://stabilizerapi.documentinsight.ai/api`.

The SDK supports the public API and the end-to-end workflow used by the bundled `run_me` helper.

## Install

```powershell
py -m pip install -e .
```

## Set The API Key

You can provide the API key in either of these ways.

### Option 1: `.env.local`

Create a `.env.local` file in the project root:

```powershell
STABILIZER_API_KEY=YOUR_STABILIZER_API_KEY
STABILIZER_PROVIDER_API_KEY=YOUR_PROVIDER_API_KEY # when using BYOK
```

### Option 2: Terminal Session Variable

PowerShell:

This is required for authenticated commands:
```powershell
$env:STABILIZER_API_KEY = "YOUR_STABILIZER_API_KEY"
```

If you want the workflow config to use BYOK (in addition to the above):

```powershell
$env:STABILIZER_PROVIDER_API_KEY = "YOUR_PROVIDER_API_KEY"
```
`STABILIZER_PROVIDER_API_KEY` is optional. Set it only when you want the workflow config to use BYOK for the provider (currently only openrouter is supported)


## Run The Workflow

Use the `run_me` module when you want the full config -> optimize -> compile -> extract flow.

Once `STABILIZER_API_KEY` is set via `.env.local` or your terminal session (for example, `$env:STABILIZER_API_KEY = "YOUR_STABILIZER_API_KEY"` in PowerShell), you can omit `--api-key` from the command examples below. The fragment `--api-key YOUR_STABILIZER_API_KEY` is optional and only needed when you want to pass the key directly on a specific command.

```powershell
py -m stabilizer_python_sdk.run_me `
  --state-file .\temp_db\run_me\2026-04-14-08-30-45.json `
  --temp-db-dir .\temp_db `
  --compile-payload-file .\compile-input.json `
  --extract-payload-file .\extract-input.json `
  --new `
  --poll-interval 2 `
  --poll-timeout 600
```

The same workflow can be controlled from code:

```python
from pathlib import Path

from stabilizer_python_sdk.run_me import RunMeSettings, run_all

state = run_all(
    settings=RunMeSettings(
        api_key="YOUR_STABILIZER_API_KEY",
        base_url="https://stabilizerapi.documentinsight.ai/api",
        temp_db_dir=Path("temp_db"),
        state_file=None,
        new_run=True,
        compile_payload_file=Path("compile-input.json"),
        extract_payload_file=Path("extract-input.json"),
        poll_interval=2.0,
        poll_timeout=600.0,
    )
)
print(state)
```

## Workflow Payload Files

Use `*-input.json` for request bodies and `*-output.json` for saved standalone responses.

`config-input.json` contains the LLM config request. Only `name` is required; `provider`, `default_model`, `compile_strength`, `base_url`, `is_default`, `byok`, and `api_key` are optional. `compile_strength` picks one of four base-model tiers (`low`, `medium`, `high`, `max`) that the server resolves to a concrete model; it defaults to `low` server-side and acts as the default for both compile and grounding when not overridden per-request. `default_model` is used only by the baseline-extraction path.

You can also pick a strength tier per request: pass `compile_options.compile_strength` on a compile and `options.grounding_strength` on an extract. Example compile fragment:

```json
"compile_options": {
  "compile_strength": "high"
}
```

Example extract fragment:

```json
"options": {
  "num_results": 3,
  "grounding_strength": "max"
}
```
`optimize-input.json` contains the prompt optimization request, including `prompt`, `json_structure`, `training_data`, and optional `optimization_options`.
`compile-input.json` contains the function creation request. Only `prompt` and `json_structure` are required; `name`, `description`, `tags`, `training_data`, `grounding_methods`, and `compile_options` are optional.
`extract-input.json` contains the extraction request, including `function_id`, `source_text`, and optional `options`, `metadata`, and `ground_truth`.

Standalone CLI runs keep their latest saved responses under `.\temp_db\general\`. For async standalone commands, `optimize`, `compile`, and `extract` update those saved responses only when you poll a job, either with `py -m stabilizer_python_sdk poll` or by passing `--poll`. The workflow state used by `run_me` is stored separately under `.\temp_db\run_me\`.

The files included in this repository are ready to use as examples:

```powershell
py -m stabilizer_python_sdk config --payload-file .\config-input.json # example config
py -m stabilizer_python_sdk.run_me --compile-payload-file .\compile-input.json --extract-payload-file .\extract-input.json # example workflow
```

## Standalone Sequence

The standalone flow is easiest to follow when you treat each module output as the source of the next module's IDs.

1. Create the config and save the latest response to `.\temp_db\general\config-output.json`.
```powershell
py -m stabilizer_python_sdk config --payload-file .\config-input.json
```

2. [Optional] Optimize the prompt, using the `config_id` from `config-output.json`. Polling writes the latest response to `.\temp_db\general\optimize-output.json`.
```powershell
py -m stabilizer_python_sdk optimize --payload-file .\optimize-input.json --config cfg_123 --poll
```

If you want that optimize step to update a compile payload automatically, pass `--alter-compile` with the compile payload path while using `--poll`. When the target compile payload does not include `compile_options.num_prompt_variations`, the CLI writes the optimize results into `compile_options.optimized_prompts`. If `num_prompt_variations` is present, the CLI keeps the count-based shape and replaces the top-level `prompt` with the first optimized prompt.
```powershell
py -m stabilizer_python_sdk optimize --payload-file .\optimize-input.json --config cfg_123 --poll --alter-compile .\compile-input.json
```

3. Compile the function, again using the `config_id` from `config-output.json`. Polling writes the latest response to `.\temp_db\general\compile-output.json`.
```powershell
py -m stabilizer_python_sdk compile --payload-file .\compile-input.json --config cfg_123 --poll
```

4. Extract with the `function_id` from `compile-output.json`. You can either copy that value into `extract-input.json` or pass it with `--function`. Polling writes the latest response to `.\temp_db\general\extract-output.json`.
```powershell
py -m stabilizer_python_sdk extract --payload-file .\extract-input.json --function fn_123 --poll
```

## Terminal Commands

Run only the commands you need, in any order.

Commands needed for regular flow:
```powershell
py -m stabilizer_python_sdk config --payload-file .\config-input.json # create config
py -m stabilizer_python_sdk optimize --payload-file .\optimize-input.json --config cfg_123 --poll --alter-compile .\compile-input.json # optimize prompt
py -m stabilizer_python_sdk compile --payload-file .\compile-input.json --config cfg_123 --poll # compile function
py -m stabilizer_python_sdk extract --payload-file .\extract-input.json --function fn_123 --poll # extract data
py -m stabilizer_python_sdk poll --job job_123 --timeout 600 # poll job (not needed when using --poll in commands above)
```

Other commands:
```powershell
py -m stabilizer_python_sdk health # check if the service is alive
py -m stabilizer_python_sdk models # list available models
py -m stabilizer_python_sdk org # get organization info
py -m stabilizer_python_sdk org-update --payload-file .\org-update.json # update organization info
py -m stabilizer_python_sdk api-keys # list api keys
py -m stabilizer_python_sdk api-key-create --payload-file .\api-key-create.json # create api key
py -m stabilizer_python_sdk api-key-revoke --key key_123 # revoke api key
py -m stabilizer_python_sdk configs # list configs
py -m stabilizer_python_sdk config-update --config cfg_123 --payload-file .\config-update.json # update config
py -m stabilizer_python_sdk config-delete --config cfg_123 # delete config
py -m stabilizer_python_sdk functions --name Invoice --tag billing # list compiled functions
py -m stabilizer_python_sdk function --function fn_123 # get compiled function info
py -m stabilizer_python_sdk function-update --function fn_123 --payload-file .\function-update.json # update compiled function metadata
py -m stabilizer_python_sdk function-delete --function fn_123 # delete compiled function
py -m stabilizer_python_sdk extractions # list previous extractions
py -m stabilizer_python_sdk job --job job_123 # get job info
py -m stabilizer_python_sdk usage --from 2026-04-01 --to 2026-04-15 # get usage info
py -m stabilizer_python_sdk state latest # get current state summary
```


Notes:

All authenticated schema-backed commands require an API key from `--api-key`, `STABILIZER_API_KEY`, or `.env.local`.

Commands that create or update resources with request bodies require an explicit `--payload-file`. The only environment-backed defaults for standalone runs are `STABILIZER_API_KEY` and `STABILIZER_PROVIDER_API_KEY`.

`state latest` reads the current standalone state summary from `.\temp_db\general\`, using `config-output.json`, `optimize-output.json`, `compile-output.json`, and `extract-output.json`.

## SDK Surface

```python
from stabilizer_python_sdk import StabilizerClient
```

`StabilizerClient` covers the public, org, runtime, management, and workflow routes.

The SDK is synchronous and dependency-free. It uses the Python standard library HTTP stack.

## Examples

```python
from stabilizer_python_sdk import StabilizerClient

client = StabilizerClient(api_key="YOUR_STABILIZER_API_KEY")
print(client.health())
print(client.supported_models())
```
