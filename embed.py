# embed.py
import os
import traceback
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-mpnet-base-v2"

def get_embedding_model():
    model = SentenceTransformer(EMBED_MODEL_NAME)
    return model

def create_chroma_client(chroma_dir="data/chroma_db"):
    """
    Create a Chroma client compatible with multiple chromadb versions.
    Preferred: chromadb.PersistentClient(path=...)
    Fallbacks: chromadb.Client(Settings(...)) or chromadb.Client()
    """
    import chromadb
    os.makedirs(chroma_dir, exist_ok=True)
    # Try PersistentClient first
    try:
        if hasattr(chromadb, "PersistentClient"):
            client = chromadb.PersistentClient(path=chroma_dir)
            print("Created chromadb.PersistentClient")
            return client
    except Exception as e:
        print("PersistentClient failed:", e)

    # Try Settings-based client (older/newer)
    try:
        from chromadb.config import Settings
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=chroma_dir))
        print("Created chromadb.Client(Settings(...))")
        return client
    except Exception as e:
        print("Client(Settings(...)) failed:", e)

    # Last resort: default client()
    try:
        client = chromadb.Client()
        print("Created chromadb.Client() fallback")
        return client
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(f"Failed to create any chroma client. Errors: {e}\nTraceback:\n{tb}")

def upsert_chunks_to_chroma(client, collection_name, chunks, model):
    """
    Upsert chunks (list of dicts) into a collection. Tolerant to API differences.
    """
    # create collection if missing
    try:
        existing = [c.name for c in client.list_collections()]
    except Exception:
        # some clients may not have list_collections(); try to create directly
        existing = []
    if collection_name not in existing:
        try:
            collection = client.create_collection(name=collection_name)
        except Exception:
            # if create_collection fails, try get_collection (some APIs auto-create)
            try:
                collection = client.get_collection(name=collection_name)
            except Exception as e:
                raise RuntimeError(f"Failed to create or get collection {collection_name}: {e}")
    else:
        collection = client.get_collection(name=collection_name)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{
        "doc_id": c["doc_id"],
        "page": c["page"],
        "source": c["source"],
        "start_char": c["start_char"],
        "end_char": c["end_char"]
    } for c in chunks]

    # compute embeddings
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Add/upsert depending on API
    try:
        if hasattr(collection, "add"):
            collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings.tolist())
        elif hasattr(collection, "upsert"):
            collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings.tolist())
        else:
            raise RuntimeError("Chroma collection has no add/upsert method.")
    except Exception as e:
        raise RuntimeError(f"Failed to add embeddings to collection: {e}")

    # Try to persist: collection.persist() or client.persist()
    persisted = False
    try:
        if hasattr(collection, "persist"):
            collection.persist()
            persisted = True
    except Exception:
        persisted = False
    try:
        if not persisted and hasattr(client, "persist"):
            client.persist()
            persisted = True
    except Exception:
        persisted = persisted

    if not persisted:
        print("Warning: Chroma persistence method not available. Data may not persist across sessions.")
    return collection
