# ingest.py
import pdfplumber
import os
from typing import List, Tuple, Dict

def pdf_to_text_pages(pdf_path: str) -> List[str]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            text = p.extract_text() or ""
            pages.append(text)
    return pages

def chunk_text(text: str, max_chars: int = 800, overlap: int = 150) -> List[Tuple[str, int, int]]:
    """
    Chunk text into overlapping pieces.
    Returns list of (chunk_text, start_char, end_char).
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        piece = text[start:end].strip()
        chunks.append((piece, start, end))
        start = max(0, end - overlap)
        if end == length:
            break
    return chunks

def ingest_pdf_to_docs(pdf_path: str, doc_id: str = None) -> List[Dict]:
    """
    Read a PDF and return list of chunk dicts with metadata.
    """
    if doc_id is None:
        doc_id = os.path.basename(pdf_path)
    pages = pdf_to_text_pages(pdf_path)
    doc_chunks = []
    for page_no, page_text in enumerate(pages, start=1):
        if not page_text or page_text.strip() == "":
            continue
        page_chunks = chunk_text(page_text, max_chars=800, overlap=150)
        for i, (piece_text, start_char, end_char) in enumerate(page_chunks):
            chunk_id = f"{doc_id}_p{page_no}_c{i}"
            doc_chunks.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "page": page_no,
                "text": piece_text,
                "start_char": start_char,
                "end_char": end_char,
                "source": pdf_path
            })
    return doc_chunks

if __name__ == "__main__":
    # quick manual test (run from project root)
    sample = os.path.join("sample_data", "company_policy.pdf")
    if os.path.exists(sample):
        chunks = ingest_pdf_to_docs(sample)
        print(f"Ingested {len(chunks)} chunks from {sample}")
        if chunks:
            print(chunks[0])
    else:
        print("Run sample_data_generator.py first to create sample PDFs.")