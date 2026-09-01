# ADK 2 upgrade — what changed and why

Applied 2026-09-01. Every change below was verified against **google-adk 2.8.0**
installed in a scratch environment: the agent was actually imported and built
against 2.8.0, the dependency set was resolved with pip, and the deploy behavior
was read out of the shipped `adk deploy agent_engine` CLI rather than from docs.

Not yet verified: an end-to-end deploy into a live project. See the checklist at
the bottom.

---

## The short version

The agent code was already structurally ADK 2 clean — no `BaseAgent` subclass, no
`_run_async_impl()` override, no `SequentialAgent`/`ParallelAgent`/`LoopAgent`, no
manual event appends. None of the 2.0 breaking changes applied. What actually
needed fixing was dependencies, one moved import, the model IDs, and the fact
that nothing in the repo deployed.

## Changes

### 1. `requirements.txt` — three separate problems

| Before | After | Why |
|--------|-------|-----|
| `google-adk (>=1.18.0,<2.0.0)` | `google-adk[gcp] (>=2.8.0,<3.0.0)` | The cap blocked ADK 2 outright. The `[gcp]` extra is not optional: in 2.x the BigQuery toolset needs `google-cloud-bigquery` **and** `google-cloud-dataplex`, and plain `google-adk` installs neither — the failure is `ImportError: cannot import name 'dataplex_v1' from 'google.cloud'`. There is no `[bigquery]` extra (pip warns and ignores it); `[bigquery-analytics]` exists but omits dataplex. |
| `google-genai (>=1.50.1,<2.0.0)` | `google-genai (>=2.19,<3.0.0)` | ADK 2.8.0 requires `google-genai>=2.19,<3`. With the old pin the install fails outright with `ResolutionImpossible`. |
| `pydantic (>=2.12.4,<3.0.0)` | `pydantic (>=2.12,<3.0.0)` | Matches what ADK asks for. |
| `agent-starter-pack (>=0.20.3,<1.0.0)` | removed | A scaffolding CLI, not a runtime dependency — it would be built into every deployed image. The current equivalent is `agents-cli scaffold enhance --deployment-target agent_engine`, installed separately when you want it. |
| `google-cloud-aiplatform[agent_engines] (>=1.127.0,<2.0.0)` | floor raised to `1.148.1` | Matches what `google-adk[gcp]` requires anyway. |

Verified: `pip install -r requirements.txt` now resolves cleanly (pip settles
OpenTelemetry at 1.42.1 to satisfy ADK). Install from the file in one command —
installing these package by package produces an OpenTelemetry conflict.

### 2. `diabetes_agent/requirements.txt` — a symlink, so the root file stays canonical

`adk deploy agent_engine` installs the `requirements.txt` it finds **inside the
agent folder**; the repo-root file is never consulted. If the agent folder has
none, the CLI silently writes a minimal one containing only
`google-cloud-aiplatform[agent_engines]` and `google-adk[a2a]` — no BigQuery, no
dataplex — and the deployed agent then fails to import at runtime.

There is no working flag for this. `--requirements_file` still exists but in
2.8.0 it only prints a deprecation warning; the value is never used.

So `diabetes_agent/requirements.txt` is now a symlink to `../requirements.txt`.
The deploy stages the agent folder with `shutil.copytree(..., symlinks=False)`,
which dereferences it — verified by running that same call — so the container
gets a real file with the root content, and there is still only one dependency
list in the repo. `4_deploy.sh` recreates the symlink (or falls back to a copy)
if it ever goes missing.

### 3. `diabetes_agent/agent.py`

- BigQuery imports moved from `google.adk.tools.bigquery` to
  `google.adk.integrations.bigquery`. The old path still works in 2.8.0 but prints
  a `DeprecationWarning` on every start, which students will ask about.
- Both agents now use `MODEL = os.environ.get("AGENT_MODEL", "gemini-flash-latest")`
  instead of a hardcoded `gemini-2.5-flash`. Gemini 2.5 Pro and Flash are on the
  retirement list — the date being reported is **October 16, 2026**, about six
  weeks out — so the demo would have started failing this fall. `AGENT_MODEL` lets
  a lab pin an explicit version without editing code; `activate.sh` sets it.

### 4. `diabetes_agent/prompts.py` — project id resolution

`PROJECT_ID` was read from the environment with a `'your-project-id'` fallback.
That variable is exported by `activate.sh` and does **not** exist inside the Agent
Runtime container, so a deployed agent would have been told its data lived in a
project called `your-project-id` — no error, just wrong SQL and an apology.

It now falls back `PROJECT_ID` → `GOOGLE_CLOUD_PROJECT` → the project behind the
container's own credentials → the placeholder. `DATASET_ID` reads `BQ_DATASET` to
match `activate.sh`.

Verified both paths against ADK 2.8.0: with `PROJECT_ID` set it resolves locally,
and with only `GOOGLE_CLOUD_PROJECT` set (what the deployment provides) it
resolves to the deployed project.

### 5. `activate.sh` — now writes the agent's `.env`

Sourcing it generates `diabetes_agent/.env` from the current shell:

    GOOGLE_GENAI_USE_ENTERPRISE=1
    GOOGLE_CLOUD_PROJECT=<your project>
    GOOGLE_CLOUD_LOCATION=global
    PROJECT_ID=<your project>
    BQ_DATASET=<BQ_DATASET>
    AGENT_MODEL=<AGENT_MODEL>

This is the mechanism that carries the project id into the deployment: the deploy
reads that file and turns each line into an environment variable on the running
agent (and lifts `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` to decide where
to deploy unless `--project` / `--region` are passed, which `4_deploy.sh` does).

Correction to an earlier note: `diabetes_agent/.env` matches a `.gitignore`
pattern but was committed before that rule existed, so it **is** tracked and does
ship with a clone. The real problem was that it shipped one machine's settings and
now a deprecated flag. Generating it means it always matches the project you are
actually in. `diabetes_agent/.env.example` is committed as documentation of the
format.

Because the file is tracked, regenerating it shows up as a git modification every
time you switch projects. If that gets annoying, untrack it and let the ignore
rule do its job:

    git rm --cached diabetes_agent/.env

(While you are in there: `diabetes_agent/__pycache__/*.pyc` is also tracked from
an early commit and can go the same way.)

New variables: `GEMINI_LOCATION` (default `global`), `AGENT_REGION` (default
`us-central1`), `AGENT_MODEL`, `AGENT_DISPLAY_NAME`. `STAGING_BUCKET` is kept but
commented — `--staging_bucket` is deprecated and unused for deployment now.

Two subtleties, both verified in the 2.8.0 CLI:

- The deploy **pops** `GOOGLE_CLOUD_PROJECT` out of the variable set after using it
  to choose the deployment target, so that name does not reach the running
  container. `PROJECT_ID` is not popped — which is why `activate.sh` writes both
  and why `prompts.py` checks `PROJECT_ID` first.
- `GOOGLE_CLOUD_LOCATION` is **not** popped. It reaches the container, which is
  what we want, because it is the model location. See item 6.

### 6. Model location vs deploy region, and the enterprise flag

`GOOGLE_CLOUD_LOCATION` is now `global`, because the current Gemini models are
served globally rather than from a region. That creates a trap worth showing the
class: the ADK CLI reads `GOOGLE_CLOUD_LOCATION` from the `.env` and uses it as the
**deployment region** whenever `--region` is not passed — so a bare
`adk deploy agent_engine diabetes_agent` would try to deploy the agent to `global`,
which Agent Runtime will not accept.

Three things keep the two apart:

- `activate.sh` exports `GEMINI_LOCATION` (default `global`, written into the
  `.env` as `GOOGLE_CLOUD_LOCATION`) and `AGENT_REGION` (default `us-central1`) as
  separate variables, and prints them on labelled lines.
- `4_deploy.sh` always passes `--region="${AGENT_REGION}"`, which takes precedence.
  The deploy prints ``Ignoring GOOGLE_CLOUD_LOCATION in .env as `--region` was
  explicitly passed`` — expected and correct.
- `4_deploy.sh` refuses to run when `AGENT_REGION` is `global`, and says where
  `global` does belong.

There are now three "locations" in the demo and they are all different:
`BQ_LOCATION=US` (dataset), `GEMINI_LOCATION=global` (model),
`AGENT_REGION=us-central1` (hosted agent). The README calls this out in Step 3.

`GOOGLE_GENAI_USE_VERTEXAI` was also replaced with `GOOGLE_GENAI_USE_ENTERPRISE=1`.
The old name still works — both ADK and google-genai read it — but ADK warns
`GOOGLE_GENAI_USE_VERTEXAI is deprecated, please use GOOGLE_GENAI_USE_ENTERPRISE
instead`, and google-genai resolves a conflict between the two in favour of the
new name.

### 7. New `4_deploy.sh`

Enables the APIs, creates the Reasoning Engine service agent, grants it BigQuery
roles, sanity-checks the agent folder, and deploys. Set `AGENT_ENGINE_ID` to
redeploy over an existing instance instead of creating a second one.

The IAM step is the one that matters: a deployed agent runs as
`service-PROJECT_NUMBER@gcp-sa-aiplatform-re.iam.gserviceaccount.com`, not as you,
and that identity starts with no BigQuery access. The failure mode is
distinctive — the agent deploys, answers general questions via search, and fails
only on data questions.

### 8. `README.md`

New Steps 11–15 covering what travels with the agent, API enablement, the IAM
grants (with the gcloud commands spelled out), deploying, and testing the deployed
agent. Plus: the ADK version check now says 2.8.0, the naming table
(Agent Runtime / `agent_engine` / `reasoningEngines`), a callout listing the
deploy flags that older tutorials use and 2.x has deprecated, ten new
troubleshooting rows for the failure modes above, and a cleanup step that deletes
the deployed agent instead of leaving it running.

---

## Deliberately not changed

- **`1_setup_bq.sh` / `2_train_model.sh` hardcode `demo_diabetes`** inside the SQL
  even though `activate.sh` treats the dataset name as a variable. Changing
  `BQ_DATASET` breaks training in a way that takes a while to spot. The fix is to
  substitute after the fact, since the heredoc has to stay single-quoted (the SQL
  is full of backticks):

      bq query --use_legacy_sql=false "$(sed "s/__DATASET__/${BQ_DATASET}/g" <<'SQL'
      CREATE OR REPLACE MODEL `__DATASET__.diabetes_model`
      ...
      SQL
      )"

  Left alone for now — it is working code and the demo always uses the default.

- **`GCS_URI` points at `gs://class-demo`**, so every student project reads the
  training CSV from that one bucket. It needs to stay readable by whatever
  accounts they use: `gcloud storage buckets get-iam-policy gs://class-demo`.

- **The demo does not exercise ADK 2's graph workflows.** It is one LLM agent with
  two tools, which runs fine on 2.x but shows nothing of the release's headline
  feature. The risk assessment is a natural graph if you want a second act: fan
  out to the BigQuery prediction and a search for current guidance, join, then let
  a writer agent combine them.

## Verify before class

- [ ] Fresh venv, `pip install -r requirements.txt`, `source activate.sh`,
      `adk web` — confirm the agent loads and a dataset question really queries
      BigQuery.
- [ ] One full `source 4_deploy.sh` into a throwaway project, then ask the
      deployed agent a data question — this is what proves the service-account
      grant is sufficient.
- [ ] The Gemini 2.5 retirement date (reported as October 16, 2026) on the model
      lifecycle page, if you plan to state it in class.
- [ ] The REST `:streamQuery` example in README Step 15 against your deployment —
      the class methods are registered by the deploy, but the request shape is
      worth confirming once.
- [ ] The cleanup snippet's `client.agent_engines.list()` / `.delete()` calls
      against your installed SDK (signatures match `google-cloud-aiplatform`
      1.165.1).
