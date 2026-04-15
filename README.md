
# quick start TLDR, just run this:
```powershell
$env:STABILIZER_API_KEY = "YOUR_STABILIZER_API_KEY"
py -m stabilizer_python_sdk.run_me --new
```
# ##################################################


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
STABILIZER_PROVIDER_API_KEY=YOUR_PROVIDER_API_KEY
```

`STABILIZER_PROVIDER_API_KEY` is optional. Set it only when you want the workflow config to use BYOK for the provider.

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

## Run The Workflow

Use the `run_me` module when you want the full config -> optimize -> compile -> extract flow.

```powershell
py -m stabilizer_python_sdk.run_me `
  --api-key YOUR_STABILIZER_API_KEY `
  --state-file .\temp_db\run_me\2026-04-14-08-30-45.json `
  --temp-db-dir .\temp_db `
  --compile-payload-file .\compile.json `
  --extract-payload-file .\extract.json `
  --new `
  --poll-interval 2 `
  --poll-timeout 600
```

If `STABILIZER_API_KEY` is already set in the environment, you can omit `--api-key`.

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
        compile_payload_file=Path("compile.json"),
        extract_payload_file=Path("extract.json"),
        poll_interval=2.0,
        poll_timeout=600.0,
    )
)
print(state)
```

## Workflow Payload Files

`config.json` should contain the LLM config request, including `name`, `provider`, `default_model`, `is_default`, `byok`, and optional `api_key`.
`compile.json` should contain the function creation request, including the prompt, schema, and training data. `extract.json` should contain the extraction request, including `function_id`, `source_text`, and optional `options`.

Standalone CLI runs keep their latest saved responses under `.\temp_db\general\`. For async standalone commands, `optimize`, `compile`, and `extract` update those saved responses only when you poll a job, either with `py -m stabilizer_python_sdk poll` or by passing `--poll`. The workflow state used by `run_me` is stored separately under `.\temp_db\run_me\`.

The files included in this repository are ready to use as examples:

```powershell
py -m stabilizer_python_sdk.config --api-key YOUR_STABILIZER_API_KEY --payload-file .\config.json
py -m stabilizer_python_sdk.run_me --api-key YOUR_STABILIZER_API_KEY --compile-payload-file .\compile.json --extract-payload-file .\extract.json
```

## Terminal Commands

Run only the commands you need, in any order.

```powershell
py -m stabilizer_python_sdk health
py -m stabilizer_python_sdk models
py -m stabilizer_python_sdk config --api-key YOUR_STABILIZER_API_KEY --payload-file .\config.json
py -m stabilizer_python_sdk optimize --api-key YOUR_STABILIZER_API_KEY --payload-file .\compile.json --config cfg_123
py -m stabilizer_python_sdk compile --api-key YOUR_STABILIZER_API_KEY --payload-file .\compile.json --config cfg_123
py -m stabilizer_python_sdk extract --api-key YOUR_STABILIZER_API_KEY --payload-file .\extract.json --function fn_123
py -m stabilizer_python_sdk poll --api-key YOUR_STABILIZER_API_KEY --job job_123 --timeout 600
py -m stabilizer_python_sdk state latest
```

Notes:

`config`, `optimize`, `compile`, `extract`, and `poll` require an API key from `--api-key`, `STABILIZER_API_KEY`, or `.env.local`.

Standalone `config`, `optimize`, `compile`, and `extract` require an explicit `--payload-file`. The only environment-backed defaults for those standalone runs are `STABILIZER_API_KEY` and `STABILIZER_PROVIDER_API_KEY`.

`state latest` reads the current standalone state summary from `.\temp_db\general\`.

## SDK Surface

```python
from stabilizer_python_sdk import StabilizerClient
```

`StabilizerClient` covers the public, org, runtime, evaluation, management, and workflow routes.

The SDK is synchronous and dependency-free. It uses the Python standard library HTTP stack.

## Examples

```python
from stabilizer_python_sdk import StabilizerClient

client = StabilizerClient(api_key="YOUR_STABILIZER_API_KEY")
print(client.health())
print(client.supported_models())
```

## Development

```powershell
py -m pytest tests
```
