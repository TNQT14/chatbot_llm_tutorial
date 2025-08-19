import os, glob

# -------- helpers (load docs, chunk, embed, index) --------
def load_documents(data_dir):
    paths = sorted(glob.glob(f"{data_dir}/*.txt"))
    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(p), "text": f.read()})
    return docs