#!/usr/bin/env sh
set -eu

if [ "${APP_ENV:-development}" = "production" ]; then
  exec gunicorn app.main:app \
    --bind "0.0.0.0:${PORT:-8000}" \
    --workers "${WEB_CONCURRENCY:-2}" \
    --worker-class uvicorn.workers.UvicornWorker \
    --access-logfile - --error-logfile -
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --reload
