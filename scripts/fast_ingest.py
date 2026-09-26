import json
from src.retrieval.qdrant_store import QdrantVectorStore

def run():
    store = QdrantVectorStore()
    with open("data/processed/chunks_semantic.json", "r") as f:
        data = json.load(f)
    print(f"Indexing {len(data)} chunks...")
    store.index_chunks(data)
    print("Done!")

if __name__ == "__main__":
    run()