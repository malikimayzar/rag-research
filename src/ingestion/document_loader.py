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
    import unicodedata

    # normalize unicode (WAJIB)
    text = unicodedata.normalize("NFKD", text)

    # fix broken hyphen across lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # normalize newlines (jaga paragraph)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # remove standalone page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # remove figure/table captions (full line)
    text = re.sub(r'(Table|Figure)\s+\d+.*', '', text)

    # spacing normalization
    text = re.sub(r'[ \t]+', ' ', text)
    # remove author lines (heuristic)
    text = re.sub(r'.*University.*\n', '', text)
    text = re.sub(r'.*Facebook AI Research.*\n', '', text)
        # remove symbol-heavy lines
    text = re.sub(r'.*[†‡⋆].*\n', '', text)

    return text.strip()

def load_pdf(filepath: str, tier: int) -> Optional[Document]:
    path = Path(filepath)
    try:
        doc = fitz.open(filepath)
        pages_text = []

        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages_text.append(text)

        cleaned_pages = []
        for page_text in pages_text:
            page_text = remove_table_noise(page_text)
            page_text = clean_text(page_text)
            if page_text:
                cleaned_pages.append(page_text)

        full_text = "\n\n".join(cleaned_pages)
        sections = extract_sections(full_text)

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
                "section_titles": [s["section_name"] for s in sections],
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )

    except Exception as e:
        print(f"[ERROR] PyMuPDF failed for {filepath}: {e}")
        return None
    
def remove_table_noise(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        digit_ratio = sum(c.isdigit() for c in line) / len(line)

        # lebih konservatif (jangan buang terlalu banyak)
        if digit_ratio > 0.8 and len(line.split()) < 4:
            continue

        if len(line) < 3:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

# remove lines dengan banyak simbol aneh
def remove_symbol_noise(text):
    lines = text.split("\n")
    clean = []
    
    for line in lines:
        symbol_ratio = sum(not c.isalnum() and not c.isspace() for c in line) / max(len(line),1)
        
        if symbol_ratio > 0.3:
            continue
            
        clean.append(line)
    
    return "\n".join(clean)

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

def extract_sections(text: str) -> list[dict]:
    # Pattern lebih kuat buat Arxiv
    patterns = [
        r'^(Abstract)$',
        r'^([0-9]+\.?\s+[A-Z\s]{3,})$',   # 1 INTRODUCTION atau 1. INTRODUCTION
        r'^((?:I|V|X)+\.\s+[A-Z\s]{3,})$', # I. INTRODUCTION
        r'^(References|REFERENCES|Bibliography)$',
        r'^(Conclusion|CONCLUSION|Conclusions)$',
        r'^(Methodology|METHODS|Proposed Method)$'
    ]
    
    lines = text.split('\n')
    sections = []
    current_section = "Abstract" # Start with Abstract for Arxiv
    current_content = []

    for line in lines:
        clean_line = line.strip()
        is_header = False
        
        for p in patterns:
            if re.match(p, clean_line, re.IGNORECASE):
                # Save previous section if it has content
                content_str = "\n".join(current_content).strip()
                if content_str:
                    sections.append({"section_name": current_section, "text": content_str})
                
                current_section = clean_line.upper() # Standardize headers
                current_content = []
                is_header = True
                break
        
        if not is_header:
            current_content.append(line)
            
    sections.append({"section_name": current_section, "text": "\n".join(current_content).strip()})
    return [s for s in sections if len(s["text"]) > 20]

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    docs = []

    for filepath in sorted(raw_dir.glob("*")):
        print(f"[DEBUG] Found file: {filepath.name}")
        stem = filepath.stem
        try:
            tier = int(stem.split("_")[0].replace("tier", ""))
        except (ValueError, IndexError) as e:
            print(f"[ERROR] Nama file tidak sesuai konvensi: {filepath.name} | {e}")
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