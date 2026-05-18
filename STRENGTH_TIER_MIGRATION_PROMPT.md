# Migration prompt: align SDK with strength-tier API change

Paste this entire file as the first prompt to Claude in this SDK project. It is self-contained — Claude has not seen the conversation that produced these server changes.

---

## Your task

The server-side API (`StableAI` repo) has been changed. This SDK still reflects the OLD API surface for `compile_options` and `extract options`, and adds nothing for the new fields on `llm-configs`. Update this SDK so its dataclasses, helpers, JSON input fixtures, tests, OpenAPI schema, README, and any `run_me` examples match the new API contract. Use the TDD workflow this project follows (red → green → refactor): write or update tests before changing source, run the full test suite at the end, and do not weaken a failing test just to make it pass.

The new API is the source of truth. Do **not** preserve back-compat for the old `compile_model` knob — the server now returns 400 if a client sends it.

## What changed on the server

A new concept — **base-model strength tier** — replaces the per-request model-name override on compile. It also gives extract a separate knob for grounding model selection. Strength is one of four canonical lowercase values:

```
low | medium | high | max
```

Default is `"low"`. The server resolves strength → concrete provider/model ID via env vars (`STRENGTH_LOW_MODEL`, `STRENGTH_MEDIUM_MODEL`, `STRENGTH_HIGH_MODEL`, `STRENGTH_MAX_MODEL`), falling back to a built-in mapping (Gemini Flash-Lite / Flash / 3.1 Pro / Claude Opus 4.7). Clients never see the model name on the strength knobs — they only pick a tier.

### 1. `POST /v1/llm-configs` and `PATCH /v1/llm-configs/{id}` — new field `compile_strength`

- New optional request field `compile_strength` (default `"low"` server-side).
- Validated against `{low, medium, high, max}`. Invalid value → 400 with `"compile_strength"` in the error message.
- The existing `default_model` field is **unchanged semantically** but is now documented as "model used at extraction time only" — it no longer drives compile.
- The response shape for an LLM config now includes `"compile_strength": "<tier>"`.
- Backward compat for older stored configs without the field: server reads them back as `"low"`.

### 2. `POST /v1/functions` (compile) — `compile_model` REMOVED, `compile_strength` ADDED

- **BREAKING**: Sending `compile_options.compile_model` now returns HTTP 400. The error body contains the strings `"compile_model"` and `"compile_strength"`.
- New optional field `compile_options.compile_strength`. If set, it must be a valid tier. If invalid → 400.
- Resolution precedence for which model compile actually runs on:
  1. `compile_options.compile_strength` (request override)
  2. The LLM config's `compile_strength` (when `compile_options.llm_config_id` is set)
  3. Default `"low"`
- The resolved strength is persisted into the compiled function's `compile_config` (NOT mutated on the org's LLM config).

### 3. `POST /v1/extract` — `grounding_strength` ADDED, `extraction_model` now independent

- New optional field `options.grounding_strength`. Validated against `{low, medium, high, max}`. Invalid → 400 with `"grounding_strength"` in the message.
- `options.extraction_model` is unchanged but its scope is narrower now: it controls **only the baseline-extraction model** (`run_baseline_extraction: true` path). It no longer feeds the grounding wrapper.
- Resolution precedence for the grounding model:
  1. `options.grounding_strength` (request override)
  2. The compiled function's stored `compile_strength`
  3. The LLM config's `compile_strength`
  4. Default `"low"`

### 4. Supported models — added `anthropic/claude-opus-4.7`

The `is_supported_model()` check in the server now accepts `anthropic/claude-opus-4.7` (used by the `"max"` tier default). If your SDK has a copy of the supported-model list, add it.

## Concrete edits you will need to make

These were located by grep before this prompt was written. Treat them as a starting checklist, not the complete set — see "Hunt for missed references" below.

### `src/stabilizer_python_sdk/compile.py`
- `CompileOptions` dataclass (around line 28): **remove** `compile_model: str | None = None`. **Add** `compile_strength: str | None = None`.
- `as_payload()` (around line 40): drop the `compile_model` branch; add a `compile_strength` branch that emits the field when non-empty.
- `from_payload()` (around line 62): drop the `compile_model` extraction; add `compile_strength`.

### `src/stabilizer_python_sdk/extract.py`
- `ExtractOptions` dataclass (around line 28): **keep** `extraction_model`; **add** `grounding_strength: str | None = None`.
- `as_payload()` (around line 36): add a branch that emits `grounding_strength` when non-empty.
- `from_payload()` (around line 52): parse `grounding_strength` from the payload.

### `src/stabilizer_python_sdk/config.py`
- `LLMConfigRequest` dataclass (around line 22): add `compile_strength: str | None = None`.
- `as_payload()` (around line 32): emit the new field when non-empty.
- `from_payload()` (around line 48): parse the new field.
- `create_llm_config()` helper (around line 61): add `compile_strength` to the kwargs and forward.

### `src/stabilizer_python_sdk/run_me.py`
- Around line 61: the call passes `default_model=…`. Optionally add `compile_strength="…"` to demonstrate the new knob. Don't break the example.

### JSON fixtures at the project root
- `config-input.json`: add `"compile_strength": "low"` (or whichever tier you want to demo).
- `compile-input.json` / `compile-heavy-input.json`: if either contains `"compile_model"`, **remove it** and replace with `"compile_strength": "<tier>"`. (Grep found no `compile_model` here right now — verify.)
- `extract-input.json` / `extract-heavy-input.json`: optionally add `"grounding_strength": "<tier>"` under `options` to demonstrate.

### Tests
- `tests/test_workflow_modules.py` — known references:
  - line ~252: `compile_model="openai/gpt-5.4"` and surrounding payload assertion (line ~271) — rewrite to use `compile_strength="high"` (or another tier) and assert the same.
  - line ~318: `extraction_model="openai/gpt-5.4-mini"` is **still valid** (extraction_model is unchanged); decide whether you want a sibling test for `grounding_strength`.
- `tests/test_client.py`, `tests/test_cli.py`, `tests/test_cli_integration.py`, `tests/test_run_me.py`, `tests/test_payload_files.py` — these reference `default_model` and should be re-read; add new tests for `compile_strength` validation/passthrough on `LLMConfigRequest` and `ExtractOptions.grounding_strength`.

### OpenAPI schema (`stabilizer_openapi_schema.json`)
This is bundled in the SDK and likely sourced from the server. Update by hand here, but also flag to the maintainer that the server-side OpenAPI generator should be re-run to confirm.
- `LlmConfig`, `LlmConfigCreateRequest`, `LlmConfigPatchRequest` (around lines 1060-1145): add a `compile_strength` property:
  ```json
  "compile_strength": {
    "type": "string",
    "enum": ["low", "medium", "high", "max"],
    "default": "low",
    "description": "Base-model strength tier used for compile and inherited as the default for grounding at extraction time."
  }
  ```
- `CompileOptions` (around line 1146): **remove** the `compile_model` property entirely. **Add** `compile_strength` (same enum/description as above, but without `default`).
- `ExtractOptions` (around line 1342): **add** `grounding_strength`:
  ```json
  "grounding_strength": {
    "type": "string",
    "enum": ["low", "medium", "high", "max"],
    "description": "Strength tier for the grounding wrapper. Independent of `extraction_model`, which now controls only the baseline-extraction path. Defaults to the function's compile-time strength when omitted."
  }
  ```
- Update the `extraction_model` description to say it now only governs the baseline-extraction model.

### README
`README.md` line ~95 mentions `provider`, `default_model`, etc. as optional config fields — add `compile_strength` to that list and write one sentence describing the four tiers. Add a short example for both `compile_options.compile_strength` and `options.grounding_strength`.

## Hunt for missed references

Before declaring done, run all of these grep queries in the SDK project and confirm each hit is either updated or genuinely unrelated. Do this twice — once before you start, once after you think you're finished:

```
rg -n "compile_model"
rg -n "compile_strength"
rg -n "extraction_model"
rg -n "grounding_strength"
rg -n "default_model"
rg -n "claude-opus-4\."
rg -n "STRENGTH_(LOW|MEDIUM|HIGH|MAX)_MODEL"
```

`compile_model` should appear **zero times** in source after your changes. It may remain in test names that explicitly assert the new 400 behavior (e.g. `test_sdk_rejects_compile_model_field`) — that's intentional. If you find any `compile_model` references in `src/` after the edit, the migration is not complete.

## Tests to add (red-first)

Write these as failing tests before you edit source:

1. `CompileOptions(compile_strength="high").as_payload()` returns `{"compile_strength": "high"}`.
2. `CompileOptions.from_payload({"compile_strength": "max"}).compile_strength == "max"`.
3. `CompileOptions` no longer accepts a `compile_model` keyword (test should be `pytest.raises(TypeError)`).
4. `ExtractOptions(grounding_strength="high").as_payload()["grounding_strength"] == "high"`.
5. `ExtractOptions(extraction_model="x", grounding_strength="low")` round-trips both fields independently.
6. `LLMConfigRequest(name="x", compile_strength="medium").as_payload()` includes `compile_strength`.
7. `create_llm_config(client, name="x", compile_strength="high")` forwards the field to the client.
8. Workflow-module integration test: a compile payload built via the SDK includes `compile_strength` but never `compile_model`.

These tests should fail before your source edits and pass after.

## Don't do these

- Do **not** keep `compile_model` as a deprecated alias that maps to a strength. The server will 400 on it. Silent translation hides the breaking change from users.
- Do **not** validate strength values client-side beyond rejecting obviously invalid types. Let the server return 400 — the SDK shouldn't drift from the server's allowed set.
- Do **not** change `extraction_model`'s type or remove it. It is still a valid, narrower override.
- Do **not** mutate the LLM config in storage / state when a compile uses an override. The server explicitly forbids that; mirror it.

## Definition of done

- All hunt-for-missed-references greps come back clean.
- New tests pass; old tests either pass or have been deliberately updated to reflect the new contract (with a clear rationale in the diff).
- `pyproject.toml`'s test command (or `pytest`) runs green end-to-end.
- OpenAPI schema, README, and JSON fixtures reflect the new fields.
- The SDK can successfully POST a compile with `compile_options.compile_strength` and an extract with `options.grounding_strength` against the new server (manual verification optional, but at least the integration test fixture should match).

## Reference: server-side files (for context if you need to cross-check)

These live in the sibling `StableAI` repo:

- `core/base_model_strength.py` — strength resolver, `DEFAULT_STRENGTH_MODELS`, env var names.
- `api/storage/org_store.py` — `create_llm_config` / `update_llm_config` now take/persist `compile_strength`.
- `api/routes/llm_configs.py` — POST/PATCH validation for `compile_strength`.
- `api/routes/functions.py` — `_resolve_compile_strength`, rejection of `compile_options.compile_model`.
- `api/routes/extract.py` — `grounding_strength` validation and `grounding_model` resolution; `model_used` in the response is now the grounding-resolved model.
- `tests/test_base_model_strength.py`, `tests/test_org_store.py`, `tests/test_v1_routes.py`, `tests/test_extraction_model_passthrough.py` — the contract is asserted there if you need precise expected payloads.
