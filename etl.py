# etl.py
import faiss, pickle, uuid
import numpy as np
from sentence_transformers import SentenceTransformer
from data import DOCUMENTS

model = SentenceTransformer("all-MiniLM-L6-v2")

# assign stable chunk_ids
for doc in DOCUMENTS:
    if "chunk_id" not in doc:
        doc["chunk_id"] = str(uuid.uuid4())

# embeddings
texts = [doc["text"] for doc in DOCUMENTS]
embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# build FAISS (L2 or cosine)
index = faiss.IndexFlatIP(embeddings.shape[1])
faiss.normalize_L2(embeddings)
index.add(embeddings.astype("float32"))

# save
faiss.write_index(index, "index.faiss")
with open("metadata.pkl", "wb") as f:
    pickle.dump(DOCUMENTS, f)


print("Indexed", len(DOCUMENTS), "chunks")
