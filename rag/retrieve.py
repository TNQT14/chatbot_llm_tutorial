from config.settings import K


def retrieve(query, index, texts, metas, embed_model, k=K):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        results.append({"text": texts[idx], "source": metas[idx], "score": float(dist)})
    return results
