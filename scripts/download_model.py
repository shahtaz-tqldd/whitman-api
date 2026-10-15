"""
python -m scripts.download_model
"""

from sentence_transformers import SentenceTransformer

from app.modules.processing.models.embeddings import MODEL_PATH

MODEL_PATH.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save(str(MODEL_PATH))

print(f"Model saved to: {MODEL_PATH.resolve()}")
