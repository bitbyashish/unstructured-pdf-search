from fastapi import FastAPI
from pydantic import BaseModel
import faiss, pickle
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

# load index + metadata
index = faiss.read_index("index.faiss")
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
    
class Query(BaseModel):
    query: str

@app.post("/search")
def search(query: Query):
    q_vec = np.array([model.encode(query.query)])
    D, I = index.search(q_vec, k=3)
    results = [metadata[i] for i in I[0]]
    return {"query": query.query, "results": results}