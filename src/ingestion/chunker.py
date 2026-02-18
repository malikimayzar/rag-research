import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from document_loader import Document, load_documents


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


def chunk_by_size(
    doc: Document,
    chunk_size: int = 512,
    overlap: int = 64
) -> list[Chunk]:
    """
    Fixed-size chunking dengan overlap.
    Parameter utama untuk eksperimen ablation.
    """
    text = doc.text
    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if len(chunk_text) < 50:  # skip chunk terlalu pendek
            break

        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_c{idx:04d}",
            doc_id=doc.doc_id,
            text=chunk_text,
            start_char=start,
            end_char=end,
            tier=doc.tier,
            chunk_index=idx,
            total_chunks=-1,  # diisi setelah loop
            metadata={
                "chunk_size": chunk_size,
                "overlap": overlap,
                "strategy": "fixed_size"
            }
        ))

        start += chunk_size - overlap
        idx += 1

    # Update total_chunks
    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    return chunks


def chunk_by_paragraph(doc: Document, min_len: int = 100, max_len: int = 1000) -> list[Chunk]:
    """
    Paragraph-aware chunking.
    Lebih natural, tapi ukuran tidak konsisten.
    Berguna untuk eksperimen perbandingan strategi.
    """
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
                # Buffer terlalu panjang, simpan dulu
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_p{idx:04d}",
                    doc_id=doc.doc_id,
                    text=buffer[:max_len].strip(),
                    start_char=0,
                    end_char=max_len,
                    tier=doc.tier,
                    chunk_index=idx,
                    total_chunks=-1,
                    metadata={"strategy": "paragraph", "min_len": min_len, "max_len": max_len}
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
                    metadata={"strategy": "paragraph", "min_len": min_len, "max_len": max_len}
                ))
                buffer = ""
                idx += 1

    # Sisa buffer
    if buffer and len(buffer) >= 50:
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}_p{idx:04d}",
            doc_id=doc.doc_id,
            text=buffer,
            start_char=0,
            end_char=len(buffer),
            tier=doc.tier,
            chunk_index=idx,
            total_chunks=-1,
            metadata={"strategy": "paragraph", "min_len": min_len, "max_len": max_len}
        ))

    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    return chunks


def chunk_documents(
    docs: list[Document],
    strategy: str = "fixed_size",
    chunk_size: int = 512,
    overlap: int = 64
) -> list[Chunk]:
    all_chunks = []

    for doc in docs:
        if strategy == "fixed_size":
            chunks = chunk_by_size(doc, chunk_size, overlap)
        elif strategy == "paragraph":
            chunks = chunk_by_paragraph(doc)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        all_chunks.extend(chunks)
        print(f"[CHUNK] {doc.doc_id} → {len(chunks)} chunks (strategy={strategy})")

    return all_chunks


def save_chunks(chunks: list[Chunk], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = [c.to_dict() for c in chunks]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved {len(chunks)} chunks → {output_path}")


def print_stats(chunks: list[Chunk]):
    lengths = [len(c.text) for c in chunks]
    print(f"\n=== Chunk Statistics ===")
    print(f"Total chunks  : {len(chunks)}")
    print(f"Avg length    : {sum(lengths)/len(lengths):.0f} chars")
    print(f"Min length    : {min(lengths)} chars")
    print(f"Max length    : {max(lengths)} chars")
    print(f"Tier breakdown:")
    for tier in [1, 2, 3]:
        tier_chunks = [c for c in chunks if c.tier == tier]
        print(f"  Tier {tier}: {len(tier_chunks)} chunks")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    docs = load_documents("data/processed/documents.json")
    if not docs:
        print("[ERROR] Tidak ada dokumen. Jalankan document_loader.py dulu.")
        exit(1)

    # Default chunking untuk mulai
    chunks = chunk_documents(docs, strategy="fixed_size", chunk_size=512, overlap=64)
    print_stats(chunks)
    save_chunks(chunks, "data/processed/chunks_512_64.json")