import os, pandas as pd
from pathlib import Path
import chromadb, chromadb.utils.embedding_functions as efns
from groq import Groq
from dotenv import load_dotenv
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()

# ── Paths / globals ────────────────────────────────────────────────────────────
data_dir = Path(__file__).parent / "data"
departments = ["engineering", "finance", "general", "marketing", "hr"]
csv_path   = data_dir / "hr" / "hr_data.csv"          # HR is special ⇒ csv only
collection_name = "Departmental_Docs"

# ── Chroma + LLM ───────────────────────────────────────────────────────────────
chroma_client = chromadb.Client()
embed_fn = efns.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
grok_client = Groq()

md_splitter  = MarkdownTextSplitter(chunk_size=500,  chunk_overlap=20)
csv_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

# ── Loaders ────────────────────────────────────────────────────────────────────
def load_markdown_docs():
    docs = []
    for dept in departments:
        for fp in (data_dir / dept).glob("*.md"):
            raw = fp.read_text(encoding="utf-8")
            for chunk in md_splitter.split_text(raw):
                docs.append({
                    "content": chunk,
                    "metadata": {"department": dept, "source": fp.name}
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

# ── Ingest ─────────────────────────────────────────────────────────────────────
def ingest_docs(force_reload=True):
    if force_reload and collection_name in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(name=collection_name)

    if collection_name in [c.name for c in chroma_client.list_collections()]:
        print(f"ℹ️  Collection '{collection_name}' already exists"); return

    print("📥 Ingesting docs …")
    col = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embed_fn)
    # de-dupe
    docs = load_markdown_docs() + load_hr_csv()
    seen, uniq = set(), []
    for d in docs:
        key = d["content"].strip()
        if key not in seen:
            seen.add(key); uniq.append(d)
    col.add(
        documents=[d["content"] for d in uniq],
        metadatas=[d["metadata"] for d in uniq],
        ids       =[f"id_{i}" for i in range(len(uniq))]
    )
    print("✅ Ingested", len(uniq), "chunks.")

# ── RAG helpers ────────────────────────────────────────────────────────────────
def get_relevant(query, allowed):
    col = chroma_client.get_collection(name=collection_name)
    return col.query(query_texts=[query], n_results=5, where={"department": {"$in": allowed}})

def rag_chain(query, role):
    allowed = [role, "general"]
    res  = get_relevant(query, allowed)
    meta, docs = res["metadatas"][0], res["documents"][0]
    if not docs:
        return "I don't know based on the current information."

    ctx = "\n---\n".join(f"[{meta[i]['source']}] {docs[i]}" for i in range(len(docs)))
    return generate_answer(query, ctx)

def generate_answer(query, context):
    prompt = f"""You are an AI assistant for FinSolve Technologies, providing role-specific information to users. 
You're responding to a user with the role:.

Follow these guidelines:
1. Answer questions based ONLY on the context provided below
2. If the information isn't in the context, say "I don't have that information" - DO NOT make up answers
3. Keep responses professional, clear, and concise
4. Include citations to the source documents when appropriate using [Document Title]
5. Focus on providing factual information relevant to the user's role
6. Consider the conversation history for context
7. For CSV data, interpret the data as structured tables with headers and rows
   - Present tabular data in a readable format
   - If asked for specific data points, extract them precisely
   - For financial data, format numbers appropriately (e.g., currency symbols, decimal places)
8. For Markdown data:
   - Properly interpret headers, lists, tables, and other formatting
   - Preserve the hierarchical structure when relevant to the query
   - Recognize and properly handle code blocks or technical content

QUESTION: {query}

CONTEXT:
{context}

ANSWER:"""
    chat = grok_client.chat.completions.create(
        model=os.environ["GROQ_MODEL"],
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
