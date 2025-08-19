from config.settings import EMBED_MODEL
from sentence_transformers import SentenceTransformer

import faiss
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr

def build_index(docs, embed_model_name=EMBED_MODEL):
    print("1) Splitting documents into chunks...")
    splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
    texts = []
    metas = []
    for d in docs:
        chunks = splitter.split_text(d["text"])
        for c in chunks:
            texts.append(c)
            metas.append(d["source"])

    print(f" -> {len(texts)} chunks created. Computing embeddings (SentenceTransformer)...")
    embed_model = SentenceTransformer(embed_model_name)
    embeddings = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    print(f" -> embeddings shape {embeddings.shape}, building FAISS index (dim={dim})...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, texts, metas, embed_model