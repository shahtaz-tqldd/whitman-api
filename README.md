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

The processing output groups products by category and aggregates recognized attribute columns:

```json
{
  "total_products": 2,
  "categories": [
    {"category": "shirts", "colors": ["blue", "red"], "sizes": ["lg", "xl"]}
  ]
}
```

The current transformation is a deterministic placeholder in
`app/modules/processing/service.py` (`build_catalog`) and is the extension point for the real ML
pipeline.

Import [the Postman collection](postman/Whitman API.postman_collection.json) for ready-to-run
requests; it stores the auth token automatically after login.

## Project layout

```text
app/
  core/           settings, logging, security (JWT + password hashing)
  db/             models, base classes, async session
  modules/
    auth/         login endpoint
    processing/   file processing + history endpoints, catalog builder
  schemas/        shared schemas (pagination)
  services/       R2 storage with local fallback
scripts/          create_user CLI
docker/           Dockerfile, Compose file, entrypoint
```

## Quality checks

```bash
ruff check .
python -m compileall app scripts
pytest
```
