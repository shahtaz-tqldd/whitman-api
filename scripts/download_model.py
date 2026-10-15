"""
python -m scripts.download_model
"""
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

from whitman.embeddings import MODEL_PATH


MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
MODEL_MARKER = MODEL_PATH / ".source_model"


def main() -> None:
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    if MODEL_MARKER.exists() and MODEL_MARKER.read_text().strip() == MODEL_NAME:
        print(f"Model already downloaded: {MODEL_NAME}")
        print(f"Model path: {MODEL_PATH.resolve()}")
        return

    # Preserve an existing model if it was saved by the previous script.
    if (MODEL_PATH / "modules.json").exists() and not MODEL_MARKER.exists():
        print(f"Existing local model found: {MODEL_PATH.resolve()}")
        MODEL_MARKER.write_text(MODEL_NAME)
        return

    print(f"Downloading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.save(str(MODEL_PATH))
    MODEL_MARKER.write_text(MODEL_NAME)

    print(f"Model saved to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
    