
from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model_cache" / "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Local embedding model not found: {MODEL_PATH}"
        )

    return SentenceTransformer(
        str(MODEL_PATH),
        device="cpu",
        local_files_only=True,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    embeddings = get_embedding_model().encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings.tolist()