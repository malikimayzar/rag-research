import json
import re
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import fitz

@dataclass
class Document:
    doc_id: str
    text: str
    source: str
    tier: int          
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)

def clean_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def load_pdf(filepath: str, tier: int) -> Optional[Document]:
    path = Path(filepath)
    try:
        doc = fitz.open(filepath)
        pages_text = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages_text.append(text)
        full_text = clean_text("\n\n".join(pages_text))
        
        if len(full_text) < 200:
            return None

        return Document(
            doc_id=path.stem,
            text=full_text,
            source=str(filepath),
            tier=tier,
            metadata={
                "pages": len(doc),
                "char_count": len(full_text),
                "filename": path.name,
                "engine": "pymupdf",
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )
    except Exception as e:
        print(f"[ERROR] PyMuPDF failed for {filepath}: {e}")
        return None

def load_text(filepath: str, tier: int) -> Optional[Document]:
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] File tidak ditemukan: {filepath}")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            full_text = f.read()

        return Document(
            doc_id=path.stem,
            text=full_text,
            source=str(filepath),
            tier=tier,
            metadata={
                "char_count": len(full_text),
                "filename": path.name
            }
        )
    except Exception as e:
        print(f"[ERROR] Gagal load {filepath}: {e}")
        return None

def save_documents(docs: list[Document], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = [doc.to_dict() for doc in docs]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved {len(docs)} documents → {output_path}")

def load_documents(input_path: str) -> list[Document]:
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs = []
    for d in data:
        docs.append(Document(
            doc_id=d["doc_id"],
            text=d["text"],
            source=d["source"],
            tier=d["tier"],
            metadata=d["metadata"]
        ))
    return docs

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    docs = []

    for filepath in sorted(raw_dir.glob("*")):
        stem = filepath.stem
        try:
            tier = int(stem.split("_")[0].replace("tier", ""))
        except (ValueError, IndexError):
            print(f"[SKIP] Nama file tidak sesuai konvensi: {filepath.name}")
            print("       Gunakan format: tier1_namafile.pdf")
            continue

        if filepath.suffix == ".pdf":
            doc = load_pdf(str(filepath), tier)
        elif filepath.suffix == ".txt":
            doc = load_text(str(filepath), tier)
        else:
            print(f"[SKIP] Format tidak didukung: {filepath.suffix}")
            continue

        if doc:
            docs.append(doc)
            print(f"[LOAD] {filepath.name} | tier={tier} | {doc.metadata['char_count']:,} chars")

    if docs:
        save_documents(docs, "data/processed/documents.json")
    else:
        print("[WARNING] Tidak ada dokumen yang berhasil diload.")
        print("       Letakkan PDF di data/raw/ dengan format: tier1_nama.pdf")