# PDF Semantic Search API

This project implements a **semantic search API** over PDF content (paragraphs, tables, and images) using **Sentence Transformers** and **FAISS** for vector search. The API is secured with **JWT Bearer tokens**.

---

## Features

- ETL pipeline to ingest and embed PDF content
- Semantic search using **Sentence Transformers**
- Vector index using **FAISS**
- Secure search API with **JWT authentication**
- Supports paragraphs, tables, and image captions
- Example synthetic dataset for demo

---

## Demo Dataset

The included `documents` list contains sample content:

```python
documents = [
    {"id": "p1", "type": "paragraph", "text": "Fraud detection in finance uses machine learning."},
    {"id": "t1", "type": "table", "text": "Year | Revenue\n2023 | 2M\n2024 | 4M"},
    {"id": "i1", "type": "image", "text": "Chart showing revenue growth from 2020 to 2024"}
]
```

---

## Setup

1. Clone the repository:
```bash
git clone <repo_url>
cd pdf-search
```

2. Create a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # Linux/macOS
myenv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ETL Pipeline

Run `etl.py` to:
- Generate embeddings for each PDF chunk
- Normalize embeddings
- Build a FAISS index
- Save metadata

```bash
python etl.py
```

---

## Running the API

Start FastAPI:
```bash
uvicorn app:app --reload
```

Access Swagger docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Generate Token

For demo purposes, generate a JWT token:
```bash
curl -X POST http://127.0.0.1:8000/token
```

---

## Search Endpoint

Send a POST request with the Bearer token:
```bash
curl -X POST http://127.0.0.1:8000/search \
    -H "Authorization: Bearer <JWT_STRING>" \
    -H "Content-Type: application/json" \
    -d '{"query": "revenue growth in 2024", "k": 3}'
```

---

## Notes

- FAISS uses cosine similarity (inner product + L2 normalization).
- JWT tokens are issued without authentication for demo purposes. For production, integrate proper login/auth systems.
- Images are converted to searchable captions; actual image search can be added with CLIP or BLIP models.

---

## Dependencies

- fastapi
- uvicorn
- faiss-cpu
- sentence-transformers
- numpy
- PyJWT
- pydantic

---

## License

MIT License

