# lifecoach_rag.py

from config.settings import DATA_DIR, EMBED_MODEL, MODEL_NAME
from data.loader import load_documents
from frontend.gradio_app import create_app
from rag.indexer import build_index
# from transformers import AutoTokenizer, AutoModelForCausalLM
import torch






if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents(DATA_DIR)
    if len(docs) == 0:
        raise SystemExit("Không có documents trong data/rag_docs — hãy thêm file .txt và chạy lại.")

    index, texts, metas, embed_model = build_index(docs)
    print("Index built. Starting Gradio app...")
    create_app(index, texts, metas, embed_model)
