from src.generation.generator import GroqGenerator
from src.api.config import settings
from src.retrieval.hybrid_retriever import MasterHybridRetriever
from src.retrieval.qdrant_store import QdrantVectorStore

def main():
    print("=" * 60)
    print("  RAG Research — Live Demo")
    print("  Hybrid Retrieval + BGE Reranker + Groq LLM")
    print("=" * 60)

    store = QdrantVectorStore()
    retriever = MasterHybridRetriever(
        vector_store=store,
        bm25_chunks_path=settings.bm25_chunks_path,
        rrf_k=settings.hybrid_rrf_k,
    )
    gen = GroqGenerator(model=settings.groq_model)

    queries = [
        "What is Retrieval-Augmented Generation?",
        "How does hybrid search improve RAG performance?",
        "What are the limitations of large language models?",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        chunks = retriever.search(q, top_k=5)
        resp = gen.generate(q, chunks)
        print(f"A: {resp.answer}")
        print(f"Latency: {resp.latency_generation_ms} ms")
        print("-" * 60)

if __name__ == "__main__":
    main()
