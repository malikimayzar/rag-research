import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    pdf_url: str
    tier: int
    topic: str

PAPER_LIST = {
    "tier1": [
        ("2312.10997", "RAG Survey"),           
        ("2005.11401", "Original RAG Paper"),    
        ("2101.00190", "BEIR Benchmark"),        
    ],
    "tier2": [
        ("2210.11610", "ReAct"),                
        ("2212.10560", "Precise Zero-Shot"),     
        ("2112.09118", "InstructGPT"),         
    ],
    "tier3": [
        ("1706.03762", "Attention Is All You Need"),  
        ("2307.09288", "Llama 2"),             
        ("2304.01852", "Instruction Tuning Survey"), 
    ]
}


def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            content = response.read().decode('utf-8')

        root = ET.fromstring(content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        entry = root.find('atom:entry', ns)
        if entry is None:
            return None

        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        published = entry.find('atom:published', ns).text[:10]

        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None:
                authors.append(name.text)

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        return {
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "published": published,
            "pdf_url": pdf_url
        }

    except Exception as e:
        print(f"  [ERROR] Failed to fetch {arxiv_id}: {e}")
        return None


def download_pdf(pdf_url: str, output_path: str) -> bool:
    try:
        headers = {'User-Agent': 'RAGResearch/1.0 (academic research)'}
        req = urllib.request.Request(pdf_url, headers=headers)

        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()

        with open(output_path, 'wb') as f:
            f.write(content)

        size_mb = len(content) / (1024 * 1024)
        print(f"  [OK] Downloaded {size_mb:.1f}MB → {output_path}")
        return True

    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return False


def build_dataset(
    download_pdfs: bool = True,
    max_per_tier: int = 2
) -> list[dict]:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_meta = []

    for tier_name, papers in PAPER_LIST.items():
        tier_num = int(tier_name.replace("tier", ""))
        print(f"\n[TIER {tier_num}] Processing {len(papers)} papers...")

        for i, (arxiv_id, label) in enumerate(papers[:max_per_tier]):
            print(f"\n  Paper: {label} ({arxiv_id})")

            # Fetch metadata
            meta = fetch_arxiv_metadata(arxiv_id)
            if not meta:
                continue

            meta["tier"] = tier_num
            meta["label"] = label
            all_meta.append(meta)

            print(f"  Title: {meta['title'][:60]}...")
            print(f"  Authors: {', '.join(meta['authors'][:3])}")

            if download_pdfs:
                filename = f"tier{tier_num}_{arxiv_id.replace('.', '_')}.pdf"
                output_path = raw_dir / filename

                if output_path.exists():
                    print(f"  [SKIP] Already downloaded: {filename}")
                else:
                    print(f"  Downloading PDF...")
                    download_pdf(meta["pdf_url"], str(output_path))
                    
                time.sleep(3)

    # Simpan metadata
    meta_path = "data/processed/dataset_meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Metadata saved → {meta_path}")
    print(f"[OK] Total papers: {len(all_meta)}")

    return all_meta


def create_adversarial_queries(meta_list: list[dict]) -> list[dict]:
    adversarial = []
    
    adversarial.extend([
        {
            "query": "What is the training cost in dollars for GPT-4?",
            "type": "out_of_corpus",
            "expected_behavior": "honest_abstention",
            "difficulty": "easy"
        },
        {
            "query": "Who won the 2024 Nobel Prize in Physics?",
            "type": "out_of_corpus",
            "expected_behavior": "honest_abstention",
            "difficulty": "easy"
        }
    ])

    adversarial.extend([
        {
            "query": "How do the attention mechanisms described across different papers compare?",
            "type": "cross_document",
            "expected_behavior": "partial_context",
            "difficulty": "hard"
        },
        {
            "query": "What are the contradictions between RAG and fine-tuning approaches?",
            "type": "contradiction_seeking",
            "expected_behavior": "partial_context",
            "difficulty": "hard"
        }
    ])

    adversarial.extend([
        {
            "query": "chunking",
            "type": "underspecified",
            "expected_behavior": "misleading_similarity",
            "difficulty": "medium"
        },
        {
            "query": "What does the paper say about performance?",
            "type": "ambiguous_reference",
            "expected_behavior": "misleading_similarity",
            "difficulty": "medium"
        }
    ])

    # Simpan
    output_path = "data/adversarial/adversarial_queries.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(adversarial, f, indent=2, ensure_ascii=False)

    print(f"[OK] {len(adversarial)} adversarial queries → {output_path}")
    return adversarial


if __name__ == "__main__":
    import sys

    print("=== RAG Research Dataset Builder ===\n")
    download = "--no-download" not in sys.argv

    if download:
        print("Mode: Download PDFs dari ArXiv")
        print("Estimasi waktu: 5-15 menit tergantung koneksi\n")
        meta = build_dataset(download_pdfs=True, max_per_tier=2)
    else:
        print("Mode: Metadata only (skip PDF download)\n")
        meta = build_dataset(download_pdfs=False, max_per_tier=2)

    print("\n=== Building Adversarial Query Set ===")
    queries = create_adversarial_queries(meta)

    print(f"\n=== Dataset Summary ===")
    for tier in [1, 2, 3]:
        tier_papers = [m for m in meta if m["tier"] == tier]
        print(f"Tier {tier}: {len(tier_papers)} papers")
    print(f"Adversarial queries: {len(queries)}")