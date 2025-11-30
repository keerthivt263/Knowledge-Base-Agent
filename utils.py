# utils.py
import html
from typing import Dict

def make_highlighted_html(chunk_text: str, query_terms: list = None):
    html_text = html.escape(chunk_text).replace("\n", "<br>")
    if query_terms:
        for t in set(query_terms):
            if not t.strip():
                continue
            escaped = html.escape(t)
            html_text = html_text.replace(escaped, f"<mark>{escaped}</mark>")
    return html_text

def format_metadata(metadata: Dict):
    return f"Source: {metadata.get('source')} — Page: {metadata.get('page')}"
