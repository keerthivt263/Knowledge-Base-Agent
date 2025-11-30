# 📚 Knowledge Base Agent 

🚀 **Live Demo (Streamlit):**  
https://your-streamlit-app-url-here

A fully local, privacy-friendly Knowledge Base Agent that lets you upload PDFs, index them into a vector database, and ask natural language questions. The system retrieves top-K relevant chunks, highlights document sources, and can optionally generate answers using an offline model.  
Everything runs locally — no API keys or paid services required.

---

## 🚀 Features

### 🔍 Semantic Search
- Upload PDFs or ingest sample documents.
- The system extracts text, chunks it, embeds it, and stores it in ChromaDB.
- Uses semantic similarity instead of keyword search.

### 🧠 Local Answer Generation (Optional)
- Uses a local FLAN-T5 model to generate concise answers based purely on retrieved chunks.
- Runs fully offline.

### 📄 Source Transparency
- Each retrieved chunk displays:
  - PDF source name  
  - Page number  
  - Distance score
- “Show Source Text” button opens the full original page.

### 📊 Vector Space Visualization
- Uses PCA to visually compare:
  - Query embedding  
  - Top-K retrieved embeddings

### 📤 Upload Any PDF
- Drag & drop documents into the sidebar.
- They are automatically indexed and stored in the local vector database.

### 🗂️ Sample Document Generation
- Auto-generates sample PDFs (policy documents, manuals, HR guides, etc.)
- Useful for testing and demos.

### 💾 Local Persistence
- Uses ChromaDB for local storage of embeddings.
- Works across multiple Chroma versions with fallback handling.

### ⚙️ Fully Offline
- Embeddings by SentenceTransformers  
- Vector DB by Chroma  
- Generation by FLAN-T5  
- UI by Streamlit  
- No internet or external APIs required.

---

## 🧩 How It Works

### 1️⃣ PDF Ingestion
- PDFs → pages → extracted text → chunked text  
- Each chunk gets metadata (PDF name, page number, character span)

### 2️⃣ Embedding Creation
- All chunks converted to embeddings using SentenceTransformers.

### 3️⃣ Vector Database Storage
- Embeddings + metadata stored in ChromaDB.
- Enables fast semantic search.

### 4️⃣ Querying
- User enters a question.
- The question is embedded and compared to database vectors.
- Retrieves top-K semantically nearest chunks.

### 5️⃣ Optional Answer Generation
- Retrieved chunks passed to FLAN-T5.
- Generates a clean, concise answer based only on the evidence.

---

## 🛠️ Technologies Used

| Component | Tool / Library |
|----------|----------------|
| UI | Streamlit |
| PDF Extraction | pdfplumber |
| Embeddings | SentenceTransformers (all-mpnet-base-v2) |
| Vector DB | ChromaDB |
| Answer Generation | google/flan-t5-small |
| Visualization | PCA (scikit-learn), Matplotlib |
| Sample PDFs | FPDF |
| Environment | Conda (Windows-friendly) |

---

### 1. Create environment
```bash
conda create -n kbagent python=3.10 -y
conda activate kbagent

### 2. Install PyTorch (CPU)
conda install pytorch cpuonly -c pytorch -y

### 3. Install dependencies
pip install -r requirements.txt
