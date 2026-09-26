import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from src.ingestion.document_loader import Document, load_documents

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    tier: int
    chunk_index: int
    total_chunks: int
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEMANTIC_MAX_CHARS = 1200  # hard ceiling — never exceed this
SEMANTIC_MIN_CHARS = 150  # minimum to avoid tiny noise chunks
# ---------------------------------------------------------------------------
# Reference detector
# ---------------------------------------------------------------------------

def is_reference(text: str) -> bool:
    t = text.lower()
    
    # Hitung berapa banyak citation bracket di teks
    bracket_matches = re.findall(r'\[\d+\]', text)
    
    # Body text punya 1-2 citation, reference list punya banyak
    # Threshold: lebih dari 3 bracket dalam satu chunk = reference list
    too_many_brackets = len(bracket_matches) > 3
    
    # Hard signals — kalau ada ini pasti reference
    has_doi = "doi:" in t
    has_arxiv = "arxiv:" in t  # lebih spesifik dari "arxiv"
    has_url = bool(re.search(r'https?://', text))
    
    return has_doi or has_arxiv or has_url or too_many_brackets


# ---------------------------------------------------------------------------
# Section detector  (FIX 3 — upgraded from naive [:200] check)
# ---------------------------------------------------------------------------
def detect_section(text: str, position: float = 1.0) -> str:
    head = text[:300].lower()

    if "abstract" in head:
        return "abstract"
    elif "introduction" in head:
        return "introduction"
    elif "conclusion" in head:
        return "conclusion"
    elif "references" in head or "bibliography" in head:
        return "references"
    elif is_reference(text) and position >= 0.7:         # catch inline citations anywhere
        return "references"
    return "body"

# ---------------------------------------------------------------------------
# Hard-limit splitter (safety net for FIX 1)
# ---------------------------------------------------------------------------

def _hard_split(text: str, max_chars: int = SEMANTIC_MAX_CHARS) -> list[str]:
    """
    Force-split any string that still exceeds max_chars after the main loop.
    Tries sentence boundary first; falls back to char split.
    Never drops content.
    """
    if len(text) <= max_chars:
        return [text]

    parts = []
    remaining = text

    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        # Find the last sentence boundary inside the window
        last_boundary = None
        for m in re.finditer(r'(?<=[.!?])\s+', window):
            last_boundary = m
        if last_boundary:
            cut = last_boundary.start()
            parts.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        else:
            # No boundary found — hard char cut
            parts.append(remaining[:max_chars].strip())
            remaining = remaining[max_chars:].strip()

    if remaining and len(remaining) >= 50:
        parts.append(remaining.strip())

    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def chunk_by_size(
    doc: Document,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    text = doc.text
    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if len(chunk_text) < 50:
            break

        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_c{idx:04d}",
            doc_id=doc.doc_id,
            text=chunk_text,
            start_char=start,
            end_char=end,
            tier=doc.tier,
            chunk_index=idx,
            total_chunks=-1,
            metadata={
                "chunk_size": chunk_size,
                "overlap": overlap,
                "strategy": "fixed_size",
                "section": detect_section(chunk_text),
            }
        ))

        start += chunk_size - overlap
        idx += 1

    for chunk in chunks:
        chunk.total_chunks = len(chunks)
    return chunks


def chunk_by_paragraph(doc: Document, min_len: int = 100, max_len: int = 1000) -> list[Chunk]:
    paragraphs = re.split(r'\n\s*\n', doc.text)
    chunks = []
    buffer = ""
    idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        buffer = (buffer + "\n\n" + para).strip() if buffer else para

        if len(buffer) >= min_len:
            if len(buffer) > max_len:
                chunk_text = buffer[:max_len].strip()
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_p{idx:04d}",
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    start_char=0,
                    end_char=max_len,
                    tier=doc.tier,
                    chunk_index=idx,
                    total_chunks=-1,
                    metadata={
                        "strategy": "paragraph",
                        "min_len": min_len,
                        "max_len": max_len,
                        "section": detect_section(chunk_text),
                    }
                ))
                buffer = buffer[max_len:]
                idx += 1
            else:
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_p{idx:04d}",
                    doc_id=doc.doc_id,
                    text=buffer,
                    start_char=0,
                    end_char=len(buffer),
                    tier=doc.tier,
                    chunk_index=idx,
                    total_chunks=-1,
                    metadata={
                        "strategy": "paragraph",
                        "min_len": min_len,
                        "max_len": max_len,
                        "section": detect_section(buffer),
                    }
                ))
                buffer = ""
                idx += 1

    if buffer and len(buffer) >= 150:
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_p{idx:04d}",
            doc_id=doc.doc_id,
            text=buffer,
            start_char=0,
            end_char=len(buffer),
            tier=doc.tier,
            chunk_index=idx,
            total_chunks=-1,
            metadata={
                "strategy": "paragraph",
                "min_len": min_len,
                "max_len": max_len,
                "section": detect_section(buffer),
            }
        ))

    for chunk in chunks:
        chunk.total_chunks = len(chunks)
    return chunks


def chunk_semantic(
    doc: Document,
    min_chars: int = SEMANTIC_MIN_CHARS,
    max_chars: int = SEMANTIC_MAX_CHARS,
) -> list[Chunk]:
    """
    Sentence-boundary-aware chunking with three hard guarantees:
      1. Never cuts mid-sentence (best effort; _hard_split handles overflows)
      2. Never produces chunks > max_chars (_hard_split safety net)
      3. Reference sentences are isolated immediately — never merged into body
    """
    # A. Split into sentences
    text = doc.text.strip()
    text = re.sub(r'\b(et al|fig|vs|e\.g|i\.e|cf|eq|approx|dept|prof|dr|mr|ms)\.',
            lambda m: m.group().replace('.', '<!DOT!>'), text, flags=re.IGNORECASE)
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.replace('<!DOT!>', '.').strip() for s in sentences if s.strip()]
    # B. Group into raw text chunks
    raw_chunks: list[str] = []
    buffer = ""

    for sent in sentences:
        # FIX 2: Reference sentence → flush buffer, emit ref as own chunk
        if is_reference(sent):
            if buffer and len(buffer) >= min_chars:
                raw_chunks.append(buffer.strip())
                buffer = ""
            if len(sent) >= 20:
                raw_chunks.append(sent.strip())
            continue

        candidate = (buffer + " " + sent).strip() if buffer else sent

        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            # Buffer is full — flush, start fresh with current sentence
            if len(buffer) >= min_chars:
                raw_chunks.append(buffer.strip())
            buffer = sent

    # Flush remaining buffer
    if buffer and len(buffer) >= 150:
        raw_chunks.append(buffer.strip())

    # FIX 1: Hard-limit pass — guarantee no chunk exceeds max_chars
    final_texts: list[str] = []
    for raw in raw_chunks:
        if len(raw) > max_chars:
            final_texts.extend(_hard_split(raw, max_chars))
        else:
            final_texts.append(raw)

    # C. Build Chunk objects
    chunks: list[Chunk] = []
    total = len(final_texts)
    for idx, chunk_text in enumerate(final_texts):
        if not chunk_text:
            continue
        position = idx / total if total > 1 else 1.0
        chunks.append(Chunk(
        chunk_id=f"{doc.doc_id}_s{idx:04d}",
        doc_id=doc.doc_id,
        text=chunk_text,
        start_char=0,
        end_char=len(chunk_text),
        tier=doc.tier,
        chunk_index=idx,
        total_chunks=-1,
        metadata={
            "strategy": "semantic",
            "section": detect_section(chunk_text, position=position),
        }
    ))

    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    return chunks


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def chunk_documents(
    docs: list[Document],
    strategy: str = "semantic",
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    all_chunks = []

    for doc in docs:
        if strategy == "fixed_size":
            chunks = chunk_by_size(doc, chunk_size, overlap)
        elif strategy == "paragraph":
            chunks = chunk_by_paragraph(doc)
        elif strategy == "semantic":
            chunks = chunk_semantic(doc)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        all_chunks.extend(chunks)
        print(f"[CHUNK] {doc.doc_id} → {len(chunks)} chunks (strategy={strategy})")

    return all_chunks


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[Chunk], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = [c.to_dict() for c in chunks]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {len(chunks)} chunks → {output_path}")


def print_stats(chunks: list[Chunk]):
    lengths = [len(c.text) for c in chunks]
    sections: dict[str, int] = {}
    for c in chunks:
        sec = c.metadata.get("section", "unknown")
        sections[sec] = sections.get(sec, 0) + 1

    over_limit = [c for c in chunks if len(c.text) > SEMANTIC_MAX_CHARS]

    print(f"\n=== Chunk Statistics ===")
    print(f"Total chunks      : {len(chunks)}")
    print(f"Avg length        : {sum(lengths)/len(lengths):.0f} chars")
    print(f"Min length        : {min(lengths)} chars")
    print(f"Max length        : {max(lengths)} chars")
    print(f"Over {SEMANTIC_MAX_CHARS} chars    : {len(over_limit)}  ← target: 0")
    print(f"\nSection breakdown:")
    for sec, count in sorted(sections.items()):
        print(f"  {sec:<15}: {count} chunks")
    print(f"\nTier breakdown:")
    for tier in [1, 2, 3]:
        tier_chunks = [c for c in chunks if c.tier == tier]
        print(f"  Tier {tier}: {len(tier_chunks)} chunks")


def print_samples(chunks: list[Chunk], n_body: int = 3, n_ref: int = 1):
    """Print random sample chunks for manual quality inspection."""
    import random

    body_chunks = [c for c in chunks if c.metadata.get("section") != "references"]
    ref_chunks  = [c for c in chunks if c.metadata.get("section") == "references"]

    print(f"\n=== Sample Body Chunks (n={n_body}) ===")
    for c in random.sample(body_chunks, min(n_body, len(body_chunks))):
        print(f"\n  [{c.chunk_id}] section={c.metadata['section']} len={len(c.text)}")
        print(f"  {c.text[:300]}{'...' if len(c.text) > 300 else ''}")

    print(f"\n=== Sample Reference Chunk (n={n_ref}) ===")
    for c in random.sample(ref_chunks, min(n_ref, len(ref_chunks))):
        print(f"\n  [{c.chunk_id}] section={c.metadata['section']} len={len(c.text)}")
        print(f"  {c.text[:300]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    docs = load_documents("data/processed/documents.json")
    if not docs:
        print("[ERROR] Tidak ada dokumen. Jalankan document_loader.py dulu.")
        exit(1)

    chunks = chunk_documents(docs, strategy="semantic")
    print_stats(chunks)
    print_samples(chunks, n_body=3, n_ref=1)
    save_chunks(chunks, "data/processed/chunks_semantic.json")