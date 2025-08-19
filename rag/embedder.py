from sentence_transformers import SentenceTransformer

def get_embedder(model_name):
    return SentenceTransformer(model_name)

def embed_texts(embedder, texts):
    return embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
