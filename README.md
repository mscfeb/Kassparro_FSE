# KASPARRO FSE I Assessment

Python assessment system for extracting factual sentences from speaker-product answers, checking them against the supplied catalogue, and evaluating exact source spans.

## Current Status

The deterministic evaluator, typed CSV/catalogue loading, output validation, Gemini-native tool-calling adapter, bounded asynchronous request policy, FastAPI routes, SQLAlchemy persistence, and optional Langfuse integration are implemented. Fake-provider tests are used for deterministic coverage. No live Gemini calls, PostgreSQL connection, or Langfuse traces are claimed until their credentials and services are configured.

The controlled v1/v2 experiment is not complete. v2 must be selected from actual v1 development failures; it must not be invented before a genuine v1 run. Final-check runs must happen only after both prompts are frozen.

## Setup

Use Python 3.12. Create a virtual environment, activate it, then install pinned dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The local `.env` already exists in this workspace with empty secret fields. Edit it locally; do not commit it. `Settings` loads it with `python-dotenv`. The default model is `gemini-2.5-flash`; actual availability depends on the configured Gemini account/API.

## Configuration

Set `GEMINI_API_KEY` for real model requests. `DATABASE_URL` is optional for unit tests and required for PostgreSQL persistence. Use a SQLAlchemy Psycopg URL:

```text
postgresql+psycopg://USERNAME:PASSWORD@HOST:5432/DATABASE_NAME
```

Percent-encode URL-special characters in a string URL, or construct the URL with SQLAlchemy's `URL.create` API to avoid manual escaping. Initialize tables with:

```powershell
python -m scripts.init_db
```

Langfuse is enabled only when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. Trace submission is best-effort and never replaces scoring or answer outcomes.

The application rejects a non-Gemini provider, any timeout other than 15 seconds, concurrency above three, more than three model turns, or more than one retry per turn. No key values are logged.

## Run

Start the API:

```powershell
uvicorn app.api:app --reload
```

Endpoints:

- `POST /classify`
- `POST /evaluation-runs`
- `GET /evaluation-runs/{id}`

Run a genuine Gemini function-calling smoke test after setting the key:

```powershell
python -m scripts.gemini_smoke
```

Run a dataset split:

```powershell
python -m scripts.run_evaluation --version v1 --split development
```

Outputs are written under `runs/v1_development/`. Predictions are saved before the evaluator reads development gold. Check-run gold-dependent metric and correctness fields are null. Provider errors are not replaced with `unknown`.

## Tests And Audit

```powershell
python -m pytest -q
python -m scripts.audit_data
```

The evaluator uses exact `(answer_id, start_char, end_char)` matching and source-quote validation. It calculates extraction/joint precision, recall and F1, per-label scores, macro F1, coverage, unmatched/missed/wrong-label reports, and the required confusion matrix. SQLite is used only in ORM unit tests; it is not evidence that PostgreSQL connectivity works.

## Data And Evidence

The source assessment and original root CSVs are retained. Byte-identical working copies are under `data/`. The audit script prints SHA-256 hashes. Development gold is only loaded for deterministic scoring after the gold-free prediction snapshot is written.

Live requirements remain credential- and service-dependent. Do not treat fake-provider output, a package import, or an empty artifact as a completed live run. Do not tune v2 or report a score improvement without actual v1/v2 development runs on the same provider/model/settings and evaluator.

AI assistance was used to produce and review implementation changes. Verification to date is recorded in `reports/pytest-initial.txt`; refresh the final report with the test suite command below after changes:

```powershell
python -m pytest -q *> reports/pytest-final.txt
```