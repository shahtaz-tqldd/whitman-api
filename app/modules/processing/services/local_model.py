from sentence_transformers import SentenceTransformer

from app.modules.processing.models.embeddings import MODEL_PATH

model = SentenceTransformer(
    str(MODEL_PATH),
    device="cpu",
    local_files_only=True,
)
