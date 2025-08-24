# app.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import faiss, pickle, numpy as np, jwt
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

SECRET = "CHANGE_ME"
ALGO = "HS256"
app = FastAPI(title="PDF Search API")

model = SentenceTransformer("all-MiniLM-L6-v2")

# load index + metadata
index = faiss.read_index("index.faiss")
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# ---- auth helpers ----
bearer_scheme = HTTPBearer()

def create_token(sub: str, ttl_min=60):
    now = datetime.utcnow()
    payload = {"sub": sub, "exp": now + timedelta(minutes=ttl_min), "iat": now}
    return jwt.encode(payload, SECRET, algorithm=ALGO)

def require_auth(cred: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    try:
        return jwt.decode(cred.credentials, SECRET, algorithms=[ALGO])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ---- models ----
class Query(BaseModel):
    query: str
    k: int = 5

# ---- endpoints ----
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/token")
def token():
    # demo: issue token for any caller
    return {"token": create_token("demo-user")}

@app.post("/search")
def search(req: Query, user=Depends(require_auth)):
    q_vec = model.encode([req.query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec.astype("float32"), k=req.k)
    results = []
    for score, idx in zip(D[0], I[0]):
        doc = metadata[idx]
        results.append({
            "doc_id": doc.get("id"),
            "chunk_id": doc["chunk_id"],
            "type": doc["type"],
            "text": doc["text"],
            "score": float(score)
        })

    return {"query": req.query, "results": results}
