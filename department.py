import os, pandas as pd
from pathlib import Path
import chromadb, chromadb.utils.embedding_functions as efns
from groq import Groq
from dotenv import load_dotenv
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

# ── Paths
data_dir = Path(__file__).parent / "data"
departments = ["engineering", "finance", "general", "marketing", "hr"]
csv_path   = data_dir / "hr" / "hr_data.csv"          
collection_name = "Departmental_Docs"

# ── Chroma 
chroma_client = chromadb.Client()
embed_fn = efns.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
grok_client = Groq()

md_splitter  = MarkdownTextSplitter(chunk_size=500,  chunk_overlap=20)
csv_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

# ── Loaders 
def load_markdown_docs():
    docs = []
    for dept in departments:
        for folder in (data_dir / dept).glob("*.md"):
            raw = folder.read_text(encoding="utf-8")
            for chunk in md_splitter.split_text(raw):
                docs.append({
                    "content": chunk,
                    "metadata": {"department": dept, "source": folder.name}
                })
    return docs

def load_hr_csv():
    docs, df = [], pd.read_csv(csv_path)
    for _, row in df.iterrows():
        raw = "\n".join(f"{c}: {row[c]}" for c in df.columns)
        for chunk in csv_splitter.split_text(raw):
            docs.append({
                "content": chunk,
                "metadata": {"department": "hr", "source": "hr_data.csv"}
            })
    return docs

# ── Ingest docs 
def ingest_docs(force_reload=True):
    if force_reload and collection_name in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(name=collection_name)

    if collection_name in [c.name for c in chroma_client.list_collections()]:
        print(f"ℹ️  Collection '{collection_name}' already exists"); return

    print("📥 Ingesting docs …")
    col = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embed_fn)
    # de
    docs = load_markdown_docs() + load_hr_csv()
    seen, uniq = set(), []
    for d in docs:
        key = d["content"].strip()
        if key not in seen:
            seen.add(key); uniq.append(d)
    col.add(
        documents=[d["content"] for d in uniq],
        metadatas=[d["metadata"] for d in uniq],
        ids      =[f"id_{i}" for i in range(len(uniq))]
    )
    print("✅ Ingested", len(uniq), "chunks.")

# ── RAG Chain
def get_relevant(query, allowed):
    col = chroma_client.get_collection(name=collection_name)
    return col.query(query_texts=[query], n_results=5, where={"department": {"$in": allowed}})

def rag_chain(query, role):
    if role == "c_level":
        allowed = ["engineering", "finance", "general", "marketing", "hr"]
    else:
        allowed = [role, "general"]
    
    res = get_relevant(query, allowed)
    meta, docs = res["metadatas"][0], res["documents"][0]
    if not docs:
        return "I don't know based on the current information."

    ctx = "\n---\n".join(f"[{meta[i]['source']}] {docs[i]}" for i in range(len(docs)))
    return generate_answer(query, ctx, role)

def generate_answer(query, context, role):
    prompt = f"""
You are an AI assistant for FinSolve Technologies, a leading FinTech company. Your job is to assist internal employees by answering questions using secure, role-specific data from company documents.

You are currently responding to a user whose role is: **{role.upper()}**

Follow these rules carefully:

### 🔒 Access Control
1. Only provide answers based on documents that are **authorized for the user's role**.
2. If the user asks about data from departments outside their role (e.g., HR asking about Finance), respond with:
   > "You are not authorized to access that information."

3. If the data is not available in the provided context, say:
   > "I don't have that information based on the current documents."

---

### 📄 Answering Instructions
4. Use **only the context provided below** — do not make up or hallucinate information.
5. Keep the answer **clear**, **professional**, and **concise**.
6. Include **document references** in square brackets, e.g., `[finance_q3_report.md]`.

---

### 📊 Context Interpretation
7. For **Markdown documents**:
   - Understand headers, bullet points, and code blocks.
   - Maintain structure and hierarchy when useful.

8. For **CSV data**:
   - Treat the data as a table with rows and columns.
   - Format numbers properly (e.g., currency like `$1200.50`, percentages like `12.3%`).
   - Provide direct values when asked for specific numbers.

---

### 📌 Context
Below is the relevant information retrieved for this query. Use it strictly.

{context}

---

### ❓ Question
{query}

---

### 💬 Final Answer:
"""
    chat = grok_client.chat.completions.create(
        model=os.environ["GROQ_MODEL"],
        temperature=0.5,
        messages=[{"role": "user", "content": prompt}]
    )
    return chat.choices[0].message.content

# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ingest_docs(force_reload=True)
    print(rag_chain("Which quarter had the best CPA and why?", "marketing"))


# client = Client()
# print([c.name for c in client.list_collections()])
# # startup_ingest.py
# if __name__ == "__main__":
#     ingest_docs(force_reload=False)
