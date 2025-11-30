# retriever.py  (robust to different chroma versions)
def get_top_k_from_chroma(client, collection_name, query_embedding, k=3):
    """
    Query the chroma collection for top-k similar chunks.
    This version avoids requesting unsupported 'ids' in include,
    and reads ids defensively if they are present in results.
    """
    # ensure collection exists (defensive)
    try:
        names = [c.name for c in client.list_collections()]
    except Exception:
        names = []
    if collection_name not in names:
        raise RuntimeError(f"Collection '{collection_name}' does not exist. Run ingestion first.")

    collection = client.get_collection(name=collection_name)

    # Request only supported include fields
    include_fields = ["metadatas", "documents", "distances"]
    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=include_fields
        )
    except Exception as e:
        # surface the original error with a helpful hint
        raise RuntimeError(f"Chroma query failed: {e}")

    hits = []
    # results format may vary between versions; handle defensively
    docs_list = results.get("documents", [[]])
    metas_list = results.get("metadatas", [[]])
    dists_list = results.get("distances", [[]])
    ids_list = results.get("ids", [[]])  # may be absent; that's okay

    # Work with the first (and only) query
    docs = docs_list[0] if len(docs_list) > 0 else []
    metas = metas_list[0] if len(metas_list) > 0 else []
    dists = dists_list[0] if len(dists_list) > 0 else []
    ids = ids_list[0] if len(ids_list) > 0 else [None] * len(docs)

    # Build hits up to k or available length
    for i in range(min(k, len(docs))):
        hit = {
            "id": ids[i] if i < len(ids) else None,
            "text": docs[i] if i < len(docs) else "",
            "metadata": metas[i] if i < len(metas) else {},
            "distance": dists[i] if i < len(dists) else None
        }
        hits.append(hit)

    return hits