from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path
import secrets, json

from department import rag_chain, ingest_docs

# ── Load users ────────────────────────────────
users_path = Path(__file__).parent / "users.json"
users = json.loads(users_path.read_text())

# ── Initialize FastAPI ────────────────────────
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(16))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ensure Chroma Collection Exists ───────────
@app.on_event("startup")
def startup_event():
    ingest_docs(force_reload=False)  # Don't re-create if exists

# ── Dependency: Authenticated User ────────────
def get_user(request: Request):
    user_name = request.session.get("user")
    if not user_name or user_name not in users:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user_name, users[user_name]["role"]

# ── Auth: Login ───────────────────────────────
@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = users.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    request.session["user"] = username
    return {"message": "Login successful", "role": user["role"]}

# ── Auth: Logout ──────────────────────────────
@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"message": "Logout successful"}

# ── Ask Endpoint (RAG) ────────────────────────
@app.post("/ask")
def ask(query: str = Form(...), user=Depends(get_user)):
    username, role = user
    try:
        answer = rag_chain(query, role)
        return {
            "user": username,
            "department": role,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Optional Debug Endpoint ───────────────────
@app.get("/collections")
def list_collections():
    from department import chroma_client
    return {"collections": [c.name for c in chroma_client.list_collections()]}
