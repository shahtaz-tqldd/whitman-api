# Whitman

A backend for turning large, unstructured product data (csv / xlsx / json) into a structured
catalog JSON output, with JWT authentication and processing history.

## Features

- JWT based authentication with a single login endpoint (username **or** email + password)
- `POST /processing`: upload a csv/xlsx/json products file, get a structured catalog JSON back
- `GET /processing/history`: paginated processing history (file url, output, timestamp, filename,
  filesize, type); uploaded files are stored in Cloudflare R2 (local disk fallback when R2 is not
  configured)
- Async PostgreSQL access with SQLAlchemy; tables are created automatically on startup
- Docker development and production entrypoints

## Quick start with Docker

1. Create local configuration:

   ```bash
   cp .env.example .env
   ```

   Replace `SECRET_KEY` with at least 32 random characters, and add your Cloudflare R2 credentials
   (optional; without them uploads are stored under `LOCAL_STORAGE_DIR`).

2. Start the API and PostgreSQL:

   ```bash
   docker compose -f docker/docker-compose.yml up --build
   ```

3. Open Swagger UI at <http://localhost:8000/docs>. The health endpoint is `GET /health`.

4. Create a login user inside the container:

   ```bash
   docker compose -f docker/docker-compose.yml exec api \
     python -m scripts.create_user --name "Jane Doe" --username jane --email jane@example.com
   ```

## Local start

Use Python 3.11 or newer and a reachable PostgreSQL database:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Use `localhost` in `DATABASE_URL` when PostgreSQL runs locally (the default host `db` targets
Docker Compose). Tables are created on startup.

## API overview

All application routes use the `/api/v1` prefix and require a JWT token
(`Authorization: Bearer <token>`), except login and health.

| Area | Endpoint | Description |
| --- | --- | --- |
| Auth | `POST /auth/login` | Login with `{identifier, password}` where identifier is username or email; returns an access token |
| Processing | `POST /processing` | Multipart upload of a csv/xlsx/json products file; returns the structured catalog output |
| Processing | `GET /processing/history` | Paginated history: `limit` (default 20, max 100), `offset` |

### Example

```bash
# 1. Login
curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"identifier": "jane", "password": "supersecret"}'

# 2. Process a products file
curl -s -X POST http://localhost:8000/api/v1/processing \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "file=@products.csv"

# 3. List processing history
curl -s "http://localhost:8000/api/v1/processing/history?limit=20&offset=0" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

The processing output profiles the uploaded products, groups them into a category
hierarchy, and proposes attribute/commercial/inventory fields per group:

```json
{
  "total_products": 2,
  "category_count": 1,
  "subcategory_count": 0,
  "category_source_column": "category",
  "categories": [
    {
      "name": "shirts",
      "product_count": 2,
      "attributes": {
        "color": {"type": "categorical", "unit": null, "sample_values": ["blue", "red"]}
      },
      "commercial_fields": {},
      "inventory_fields": {},
      "subcategories": []
    }
  ],
  "alias_suggestions": [],
  "ml_role_discovery": {"enabled": false, "status": "disabled", "predictions": []}
}
```

The catalog builder lives in the reusable `whitman` package (see below); the extension
point for the pipeline is `whitman/pipeline.py`.

Import [the Postman collection](postman/Whitman API.postman_collection.json) for ready-to-run
requests; it stores the auth token automatically after login.

## Catalog engine (`whitman/`)

All product-processing logic is a standalone package with **no FastAPI, database or
settings dependencies**, so it can later be published and reused in any project:

```text
whitman/
  parsing.py      csv/xlsx/json uploads -> normalized row dicts
  profiler.py     column stats, inferred types and column roles
  discovery.py    category hierarchy, field buckets, alias suggestions
  embeddings.py   local sentence embeddings (model_cache/all-MiniLM-L6-v2)
  pipeline.py     build_catalog(): profile + discover in one call
  ml/             trainable column-role classifier
    features.py   profiler output -> ML feature frame
    dataset.py    labeled JSONL loading/validation + leakage-safe splits
    train.py      train and persist the classifier artifact
    evaluate.py   holdout evaluation with confidence analysis
    predict.py    role predictions with review flags
```

Prepare the local embedding model once:

```bash
python -m scripts.download_model
```

Train the column-role classifier on labeled examples (see
`whitman/ml/data/column_examples.jsonl` for the expected JSONL format):

```bash
python -m whitman.ml.train --dataset whitman/ml/data/column_examples.jsonl
```

## Project layout

```text
app/                FastAPI application (API layer)
  core/             settings, logging, security (JWT + password hashing)
  db/               SQLAlchemy models, base classes, async session
  schemas/          shared API schemas (pagination)
  services/         R2 storage with local fallback
  modules/
    auth/           login endpoint
    processing/     upload/history endpoints, service, DB repository
whitman/            reusable catalog engine (see above)
scripts/            create_user, download_model CLIs
docker/             Dockerfile, Compose file, entrypoint
postman/            ready-to-run API collection
tests/              pytest suite (API, engine, ML)
model_cache/        local embedding model (gitignored)
```

## Quality checks

```bash
ruff check .
python -m compileall app scripts
pytest
```
