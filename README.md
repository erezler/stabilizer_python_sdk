# stabilizer-python-sdk

Standalone Python SDK for the Stabilizer API at `https://stabilizerapi.documentinsight.ai/api`.

The client defaults to the production base URL and mirrors the walkthrough flow in `https://stabilizer.documentinsight.ai/api/v1/`:

1. Health check
2. Register an LLM config
3. Optimize a prompt (optional)
4. Compile a function
5. Run extraction and poll the job

## Package Surface

```python
from stabilizer_python_sdk import StabilizerAdminClient, StabilizerClient
```

`StabilizerClient` covers the public, org, runtime, evaluation, and management routes.

`StabilizerAdminClient` covers `/v1/admin/*` routes.

The SDK is synchronous and dependency-free. It uses the Python standard library HTTP stack.

## Walkthrough Example

```python
from stabilizer_python_sdk import StabilizerClient

client = StabilizerClient(api_key="YOUR_STABILIZER_API_KEY")

# Step 1: health check
print(client.health())
print(client.supported_models())

# Step 2: register an LLM config
config = client.create_llm_config(
    {
        "name": "Primary config",
        "provider": "openai",
        "api_key": "YOUR_PROVIDER_KEY",
        "default_model": "openai/gpt-5.4-mini",
        "is_default": True,
    }
)

# Step 3: optimize the prompt (optional)
optimize_job = client.optimize_prompt(
    {
        "prompt": "Extract the event details into JSON.",
        "json_structure": {
            "event_title": "string",
            "start_date": "string",
            "ticket_price_usd": "number",
            "audience_rating": "string",
            "has_live_music": "boolean",
            "sponsor_name": "string|null",
        },
        "training_data": [
            {
                "source_text": (
                    "The Midnight Market opens at Union Yard on October 3, 2026. "
                    "Admission is $12. The notice says it is suitable for all ages. "
                    "No live music is scheduled. The event is supported by Ember Bank."
                ),
                "extracted_json": {
                    "event_title": "Midnight Market",
                    "start_date": "2026-10-03",
                    "ticket_price_usd": 12,
                    "audience_rating": "general",
                    "has_live_music": False,
                    "sponsor_name": "Ember Bank",
                },
            },
            {
                "source_text": (
                    "The After Dark Film Expo will be held on November 21, 2026, at the "
                    "Old Foundry Hall. Tickets are priced at $25. Due to mature screening "
                    "themes, the listing marks the event for adults only. A live synth set "
                    "follows the final screening. No sponsor is named in the announcement."
                ),
                "extracted_json": {
                    "event_title": "After Dark Film Expo",
                    "start_date": "2026-11-21",
                    "ticket_price_usd": 25,
                    "audience_rating": "adult",
                    "has_live_music": True,
                    "sponsor_name": None,
                },
            },
            {
                "source_text": (
                    "Riverfront Makers Day is scheduled for August 9, 2026. Entry costs $7. "
                    "The flyer says the program is recommended for ages 13 and up. Acoustic "
                    "performers will play throughout the afternoon. The event is presented "
                    "by Lantern Mobile."
                ),
                "extracted_json": {
                    "event_title": "Riverfront Makers Day",
                    "start_date": "2026-08-09",
                    "ticket_price_usd": 7,
                    "audience_rating": "teen",
                    "has_live_music": True,
                    "sponsor_name": "Lantern Mobile",
                },
            },
            {
                "source_text": (
                    "Skyline Board Game Night starts on December 5, 2026, at Civic Loft. "
                    "Tickets are $9. The poster says it is open to all ages. A house DJ "
                    "will play between tournament rounds. The event is backed by Cedar "
                    "Street Bank."
                ),
                "extracted_json": {
                    "event_title": "Skyline Board Game Night",
                    "start_date": "2026-12-05",
                    "ticket_price_usd": 9,
                    "audience_rating": "general",
                    "has_live_music": True,
                    "sponsor_name": "Cedar Street Bank",
                },
            },
            {
                "source_text": (
                    "The Neon Comics Swap lands at Warehouse Annex on July 18, 2026. "
                    "Admission is $14. The listing recommends the late session for teens. "
                    "No live music will be part of the program. The event announcement "
                    "does not mention a sponsor."
                ),
                "extracted_json": {
                    "event_title": "Neon Comics Swap",
                    "start_date": "2026-07-18",
                    "ticket_price_usd": 14,
                    "audience_rating": "teen",
                    "has_live_music": False,
                    "sponsor_name": None,
                },
            },
        ],
    }
)
optimized_prompt_job = client.wait_for_job(optimize_job["job_id"], timeout=1800)

# Step 4: compile a function
compile_job = client.compile_function(
    {
        "name": "Event details extractor",
        "description": "Extracts event details from text",
        "tags": ["events", "walkthrough"],
        "prompt": (
            "Extract the event details into JSON.\n"
            "Rules:\n"
            "- `event_title`: exact title as written in the source text.\n"
            "- `start_date`: convert the event date to ISO format `YYYY-MM-DD`.\n"
            "- `ticket_price_usd`: numeric value only, with no currency symbol.\n"
            "- `audience_rating`: must be one of `general`, `teen`, or `adult`.\n"
            "- `has_live_music`: boolean only (`true` or `false`).\n"
            "- `sponsor_name`: exact sponsor name only if it is explicitly mentioned; otherwise return null.\n"
            "- Do not invent missing values.\n"
            "- Return only valid JSON matching the schema."
        ),
        "json_structure": {
            "event_title": "string",
            "start_date": "string",
            "ticket_price_usd": "number",
            "audience_rating": "string",
            "has_live_music": "boolean",
            "sponsor_name": "string|null",
        },
        "grounding_methods": [
            "hard_grounding",
            "soft_grounding",
            "intent_verification",
            "constraints_validation",
            "coverage_check",
            "fabrication_hardening",
            "coherence_overview",
            "stress_test",
        ],
        "compile_options": {
            "num_prompt_variations": 3,
        },
    }
)
compiled = client.wait_for_job(compile_job["job_id"], timeout=600)
function_id = compiled["result"]["function_id"]

# Step 5: run extraction
extract_job = client.extract(
    {
        "function_id": function_id,
        "source_text": (
            "The Harbor Lights Food Fair returns to Seabreak Plaza on September 14, 2026, "
            "for an evening of local chefs, dessert stalls, and waterfront seating. Entry "
            "costs $18 for advance tickets and includes access to the tasting court. "
            "Organizers describe the fair as suitable for ages 13 and up because the "
            "late-night program runs past 10 p.m. A live jazz trio will perform from "
            "8 p.m. to 10 p.m. The event bulletin notes that the program is presented in "
            "partnership with Northshore Credit Union."
        ),
        "options": {"num_results": 3},
    }
)
result = client.wait_for_job(extract_job["job_id"], timeout=600)
print(result["result"])
```

## Management Examples

```python
client.list_llm_configs()
client.list_functions(tag="walkthrough")
client.get_usage(from_="2026-04-01", to="2026-04-13")
client.get_job("job_123")
```

## Admin Examples

```python
from stabilizer_python_sdk import StabilizerAdminClient

admin = StabilizerAdminClient(admin_api_key="YOUR_ADMIN_KEY")
created = admin.create_org({"name": "Example Org"})
print(created["org"]["org_id"])
print(created["api_key"]["key_value"])
```

## Development

```powershell
py -m pytest tests
```
