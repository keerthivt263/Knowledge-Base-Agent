# app.py
import streamlit as st
import os
from ingest import ingest_pdf_to_docs
from embed import get_embedding_model, create_chroma_client, upsert_chunks_to_chroma
from retriever import get_top_k_from_chroma
from generator import FlanT5Generator
from utils import make_highlighted_html, format_metadata
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pdfplumber

st.set_page_config(page_title="Knowledge Base Agent", layout="wide")

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "kb_collection"

@st.cache_resource
def init_components():
    embed_model = get_embedding_model()
    chroma_client = create_chroma_client(CHROMA_DIR)
    return embed_model, chroma_client

embed_model, chroma_client = init_components()

st.title("📚 Knowledge Base Agent — Document QA")

with st.sidebar:
    st.header("Ingest documents")
    uploaded_files = st.file_uploader("Upload PDF(s)", accept_multiple_files=True, type=['pdf'])
    if st.button("Ingest uploaded PDFs"):
        if not uploaded_files:
            st.error("No files uploaded.")
        else:
            all_chunks = []
            os.makedirs("data", exist_ok=True)
            for f in uploaded_files:
                save_path = os.path.join("data", f.name)
                with open(save_path, "wb") as out:
                    out.write(f.read())
                chunks = ingest_pdf_to_docs(save_path, doc_id=f.name)
                all_chunks.extend(chunks)
            if all_chunks:
                st.info(f"Ingesting {len(all_chunks)} chunks into Chroma...")
                try:
                    upsert_chunks_to_chroma(chroma_client, COLLECTION_NAME, all_chunks, embed_model)
                    st.success("Ingestion complete. Chroma DB persisted (if supported).")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
    st.markdown("---")
    if st.button("Ingest sample docs (auto)"):
        from sample_data_generator import make_sample_pdfs
        make_sample_pdfs()
        data_dir = "sample_data"
        chunks = []
        for fname in os.listdir(data_dir):
            if fname.endswith(".pdf"):
                chunks.extend(ingest_pdf_to_docs(os.path.join(data_dir, fname), doc_id=fname))
        try:
            upsert_chunks_to_chroma(chroma_client, COLLECTION_NAME, chunks, embed_model)
            st.success("Sample docs ingested.")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

st.markdown("## Ask a question")
question = st.text_input("Enter your question here")

top_k = st.sidebar.slider("Top K results", 1, 10, 3)

col1, col2 = st.columns([2,1])

with col1:
    if st.button("Search & Answer") and question.strip():
        try:
            q_emb = embed_model.encode([question], convert_to_numpy=True)[0]
        except Exception as e:
            st.error(f"Failed to compute query embedding: {e}")
            q_emb = None

        if q_emb is not None:
            try:
                hits = get_top_k_from_chroma(chroma_client, COLLECTION_NAME, q_emb, k=top_k)
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                hits = []

            if hits:
                st.subheader("Top retrieved chunks")
                for i, h in enumerate(hits, start=1):
                    st.markdown(f"**Result {i}** — {format_metadata(h['metadata'])} — distance: {h.get('distance', 0):.4f}")
                    st.markdown(make_highlighted_html(h['text'], query_terms=[question]), unsafe_allow_html=True)
                    if st.button(f"Show source text for {h['id']}"):
                        try:
                            import os
                            src = h['metadata']['source']

                            # normalize Windows paths
                            src = os.path.normpath(src)

                            page_no = h['metadata']['page']

                            with pdfplumber.open(src) as pdf:
                                page_text = pdf.pages[page_no - 1].extract_text() or "(No text detected on this page)"
                                st.code(page_text, language="text")

                        except Exception as e:
                            st.error(f"Could not open source: {e}")

                # generator (optional)
                try:
                    gen = FlanT5Generator()
                    if gen.ready:
                        answer = gen.generate_answer(question, hits, max_length=256)
                        st.subheader("🔎 Generated Answer")
                        st.write(answer)
                    else:
                        st.info("Generation model not available; showing retrieved chunks as evidence.")
                except Exception as e:
                    st.warning("Generation failed; showing retrieved chunks. Error: " + str(e))

                # PCA visualization
                try:
                    emb_list = [embed_model.encode(h['text']) for h in hits]
                    q_vec = q_emb
                    all_vecs = np.vstack([q_vec, np.vstack(emb_list)])
                    pca = PCA(n_components=2)
                    pca_res = pca.fit_transform(all_vecs)
                    fig, ax = plt.subplots()
                    ax.scatter(pca_res[1:,0], pca_res[1:,1], label="chunks")
                    ax.scatter(pca_res[0,0], pca_res[0,1], label="query", marker='*', s=150)
                    for idx, h in enumerate(hits):
                        ax.annotate(f"#{idx+1}", (pca_res[idx+1,0], pca_res[idx+1,1]))
                    ax.set_title("PCA of query vs top chunk embeddings")
                    ax.legend()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"PCA visualization failed: {e}")
            else:
                st.info("No results found. Try ingesting documents or increasing Top-K.")
        else:
            st.error("Query embedding failed; cannot search.")
with col2:
    st.markdown("### Quick controls")
    if st.button("Clear Chroma DB (danger)"):
        import shutil
        try:
            shutil.rmtree(CHROMA_DIR)
            st.success("Chroma DB deleted.")
        except Exception as e:
            st.error(f"Failed to delete Chroma DB: {e}")
    st.markdown("### Tips")
    st.write("""
    - Upload PDFs in the sidebar and click 'Ingest uploaded PDFs' to add them to the KB.  
    - Use 'Ingest sample docs (auto)' to create demo PDFs and index them.  
    - If generation is slow or fails, rely on top-K chunks as the answer (they are evidence).
    """)
