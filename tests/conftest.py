import os

os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-at-least-32-characters")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/whitman_test")
