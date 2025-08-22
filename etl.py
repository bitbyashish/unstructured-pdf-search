import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from data import documents

model = SentenceTransformer("all-MiniLM-L6-v2")

# convert text to vectors
embeddings = np.array([model.encode(doc["text"]) for doc in documents])

# build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# save index and metadata
faiss.write_index(index, "index.faiss")

import pickle
with open("metadata.pkl", "wb") as f:
    pickle.dump(documents, f)