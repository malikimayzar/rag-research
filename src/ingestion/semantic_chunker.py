from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- IMPORT RUST ENGINE ---
try:
    import semantic_chunker_rust
except ImportError:
    raise ImportError("Library 'semantic_chunker_rust' tidak ditemukan. Jalankan 'maturin develop' di folder rust dulu!")

# Config 
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"   
BREAKPOINT_PCT   = 85         
MAX_CHUNK_TOKENS = 512        
MIN_CHUNK_TOKENS = 50         
OVERLAP_SENTENCES = 1         

# Dataclass 
@dataclass
class SemanticChunk:
    chunk_id:    str
    doc_id:      str
    text:        str
    token_count: int
    sentence_count: int
    start_sentence_idx: int
    end_sentence_idx:   int
    metadata: dict

# Utilities 
def _estimate_tokens(text: str) -> int:
    return len(text) // 4

def _compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    sims = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]
        sims.append(float(sim)) 
    return sims

# Main Chunker 
class SemanticChunker:
    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        breakpoint_percentile: int = BREAKPOINT_PCT,
        max_chunk_tokens: int = MAX_CHUNK_TOKENS,
        min_chunk_tokens: int = MIN_CHUNK_TOKENS,
        overlap_sentences: int = OVERLAP_SENTENCES,
    ):
        print(f"[SemanticChunker] Engine: RUST + PYTHON (Hybrid)")
        print(f"  - Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.breakpoint_percentile = breakpoint_percentile
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.overlap_sentences = overlap_sentences

    def chunk_document(self, doc: dict) -> List[SemanticChunk]:
        doc_id = doc.get("doc_id", "unknown")
        text   = doc.get("text", "")
        
        if not text.strip():
            print(f"  [SKIP] {doc_id} — empty text")
            return []
        sentences = semantic_chunker_rust.split_sentences_rs(text)
        
        if len(sentences) < 2:
            return [self._make_chunk(doc_id, sentences, 0, len(sentences)-1, 0, doc.get("metadata", {}))]
        embeddings = self.model.encode(sentences, batch_size=64, show_progress_bar=False)
        similarities = _compute_similarity_matrix(embeddings)
        threshold = np.percentile(similarities, 100 - self.breakpoint_percentile)
        breakpoints = semantic_chunker_rust.find_breakpoints_rs(similarities, float(threshold))
        chunk_groups = semantic_chunker_rust.assemble_chunks_rs(
            sentences, breakpoints, self.overlap_sentences
        )
        final_chunks = []
        chunk_idx = 0

        for group_sentences in chunk_groups:
            sub_chunks = self._enforce_max_tokens(group_sentences)
            for sub_sentences in sub_chunks:
                chunk = self._make_chunk(
                    doc_id, sub_sentences, 0, 0, chunk_idx, 
                    doc.get("metadata", {})
                )
                if chunk.token_count >= self.min_chunk_tokens:
                    final_chunks.append(chunk)
                    chunk_idx += 1     
        print(f"  [OK] {doc_id} → {len(sentences)} sents → {len(final_chunks)} chunks (via Rust)")
        return final_chunks

    def _enforce_max_tokens(self, sentences: List[str]) -> List[List[str]]:
        result = []
        current = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = _estimate_tokens(sent)
            if current_tokens + sent_tokens > self.max_chunk_tokens and current:
                result.append(current)
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens
        if current:
            result.append(current)
        return result

    def _make_chunk(
        self, doc_id: str, sentences: List[str],
        start_idx: int, end_idx: int, chunk_idx: int, metadata: dict
    ) -> SemanticChunk:
        text = " ".join(sentences)
        return SemanticChunk(
            chunk_id=f"{doc_id}_rs_{chunk_idx:04d}", 
            doc_id=doc_id,
            text=text,
            token_count=_estimate_tokens(text),
            sentence_count=len(sentences),
            start_sentence_idx=start_idx,
            end_sentence_idx=end_idx,
            metadata={**metadata, "engine": "rust_semantic", "chunk_idx": chunk_idx},
        )

    def chunk_all_documents(self, documents: List[dict]) -> List[SemanticChunk]:
        all_chunks = []
        t0 = time.time()
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        elapsed = time.time() - t0
        print(f"\n[SemanticChunker] Total: {len(all_chunks)} chunks dari {len(documents)} dokumen ({elapsed:.2f}s)")
        return all_chunks

# Save 
def save_chunks(chunks: List[SemanticChunk], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(c) for c in chunks]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {len(chunks)} chunks → {output_path}")

# CLI 
if __name__ == "__main__":
    import sys
    docs_path = Path("data/processed/documents.json")
    if not docs_path.exists():
        print(f"[ERROR] {docs_path} tidak ditemukan.")
        sys.exit(1)

    with open(docs_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    chunker = SemanticChunker()
    chunks = chunker.chunk_all_documents(documents)
    save_chunks(chunks, "data/processed/chunks_semantic.json")