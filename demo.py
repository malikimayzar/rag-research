from src.generation.generator import GroqGenerator
from src.retrieval.qdrant_store import QdrantVectorStore, HybridRetriever

def main():
    print("=" * 60)
    print("  RAG Research — Live Demo")
    print("  Hybrid Retrieval + BGE Reranker + Groq LLM")
    print("=" * 60)

    store = QdrantVectorStore()
    retriever = HybridRetriever(vector_store=store)
    gen = GroqGenerator()

    queries = [
        "What is Retrieval-Augmented Generation?",
        "How does hybrid search improve RAG performance?",
        "What are the limitations of large language models?",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        chunks = retriever.search(q, k=5)
        resp = gen.generate(q, chunks)
        print(f"A: {resp.answer}")
        print(f"⚡ Latency: {resp.latency_generation}s")
        print("-" * 60)

if __name__ == "__main__":
    main()
