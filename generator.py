# generator.py
import torch

GEN_MODEL = "google/flan-t5-small"

class FlanT5Generator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL).to(self.device)
            self.ready = True
        except Exception as e:
            print("Generator initialization failed (will fallback to extractive). Error:", e)
            self.ready = False

    def generate_answer(self, question: str, contexts: list, max_length=256):
        if not self.ready:
            raise RuntimeError("Generator not available.")
        context_text = "\n\n---\n\n".join([f"[source: {c['metadata']['source']} | page: {c['metadata']['page']}]\n{c['text']}" for c in contexts])
        prompt = f"Use the following extracted information from documents to answer the question. If the answer is not present, say 'I don't find a clear answer in the documents.'\n\nQUESTION: {question}\n\nCONTEXT:\n{context_text}\n\nAnswer concisely and include which source/page supports each claim."
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        out = self.model.generate(**inputs, max_length=max_length, num_beams=2, early_stopping=True)
        answer = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return answer
