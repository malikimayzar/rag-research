#!/usr/bin/env python3
"""
Audit konsistensi chunk_id antara ground truth / eval files
dengan corpus chunk yang valid saat ini (chunks_semantic.json).

READ-ONLY — script ini tidak mengubah file apa pun, hanya membaca
dan mencetak laporan ke stdout (dan opsional menyimpan JSON summary).

Cara pakai (dari root project rag-research):
    python audit_chunk_id_consistency.py

Atau simpan hasil ke file:
    python audit_chunk_id_consistency.py --save results/debug/chunk_id_audit.json
"""

from __future__ import annotations
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHUNKS_FILE = "data/processed/chunks_semantic.json"

GT_FILES = [
    "data/processed/ground_truth_qa.json",
    "data/processed/ground_truth_qa_v2.json",
    "data/processed/ground_truth_qa_clean.json",
    "data/processed/ground_truth_qa_rebuilt.json",
    "data/processed/train_eval_v2.json",
    "data/processed/holdout_eval_v2.json",
    "data/processed/train_eval.json",
    "data/processed/holdout_eval.json",
]

# candidate field names that might hold chunk id references
CHUNK_ID_FIELD_CANDIDATES = [
    "gold_chunk_id",
    "gold_chunk_ids",
    "source_chunk",
    "source_chunk_id",
    "chunk_id",
]

FOCUS_FILES = {
    "data/processed/train_eval_v2.json",
    "data/processed/holdout_eval_v2.json",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None, f"FILE NOT FOUND: {path}"
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, f"JSON PARSE ERROR: {path} -> {e}"


def extract_doc_id_from_chunk_id(chunk_id: str) -> str:
    """
    Best-effort: strip suffix like _rs_0216 or _s0143 to recover doc_id.
    Assumes doc_id is everything before the LAST chunk-suffix pattern.
    """
    if not chunk_id:
        return ""
    # patterns: _rs_0216  OR  _s0143  OR  _sub<N>
    m = re.match(r"^(.*?)(_rs_\d+|_s\d+|_sub\d+)$", chunk_id)
    if m:
        return m.group(1)
    return chunk_id


def build_valid_chunk_index(chunks_data) -> tuple[set, dict]:
    """
    Returns:
      valid_ids: set of all valid chunk_id strings currently in corpus
      doc_to_chunk_ids: dict doc_id -> list of chunk_ids belonging to it
    """
    valid_ids = set()
    doc_to_chunk_ids = defaultdict(list)

    items = chunks_data
    if isinstance(chunks_data, dict):
        # some pipelines wrap chunks under a key like "chunks"
        for key in ("chunks", "data", "items"):
            if key in chunks_data and isinstance(chunks_data[key], list):
                items = chunks_data[key]
                break

    if not isinstance(items, list):
        return valid_ids, doc_to_chunk_ids

    for c in items:
        if not isinstance(c, dict):
            continue
        cid = c.get("chunk_id") or c.get("id")
        doc_id = c.get("doc_id") or extract_doc_id_from_chunk_id(cid or "")
        if cid:
            valid_ids.add(cid)
            doc_to_chunk_ids[doc_id].append(cid)

    return valid_ids, doc_to_chunk_ids


def extract_entries(gt_data):
    """
    Normalize various ground-truth file shapes into a flat list of dict entries.
    """
    if isinstance(gt_data, list):
        return gt_data
    if isinstance(gt_data, dict):
        for key in ("data", "samples", "items", "queries", "eval_set"):
            if key in gt_data and isinstance(gt_data[key], list):
                return gt_data[key]
        # fallback: maybe dict of id -> entry
        if all(isinstance(v, dict) for v in gt_data.values()):
            return list(gt_data.values())
    return []


def get_chunk_refs(entry: dict) -> list[str]:
    """
    Pull out every chunk-id-like reference from an entry, across all
    known field name candidates. Handles both single string and list values.
    """
    refs = []
    for field in CHUNK_ID_FIELD_CANDIDATES:
        if field in entry and entry[field]:
            val = entry[field]
            if isinstance(val, list):
                refs.extend([v for v in val if isinstance(v, str)])
            elif isinstance(val, str):
                refs.append(val)
    return refs


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def audit(base_dir: str = ".") -> dict:
    base = Path(base_dir)
    chunks_path = base / CHUNKS_FILE
    chunks_data, err = load_json(str(chunks_path))
    if err:
        print(f"[FATAL] Tidak bisa load {CHUNKS_FILE}: {err}")
        return {}

    valid_ids, doc_to_chunk_ids = build_valid_chunk_index(chunks_data)
    print(f"[OK] Loaded {len(valid_ids)} valid chunk_id dari {CHUNKS_FILE}")
    print(f"[OK] {len(doc_to_chunk_ids)} unique doc_id ditemukan di corpus\n")

    results = {}
    example_pool = []  # collect (file, entry_query, old_chunk_id, doc_id, candidate_new_ids)

    print(f"{'FILE':<45} {'TOTAL':>7} {'VALID':>7} {'INVALID':>8} {'%INVALID':>9}")
    print("-" * 80)

    for rel_path in GT_FILES:
        full_path = base / rel_path
        data, err = load_json(str(full_path))
        if err:
            print(f"{rel_path:<45} {'--':>7} {'--':>7} {'--':>8} {err}")
            results[rel_path] = {"error": err}
            continue

        entries = extract_entries(data)
        total = len(entries)
        valid_count = 0
        invalid_count = 0
        invalid_samples = []

        for e in entries:
            if not isinstance(e, dict):
                continue
            refs = get_chunk_refs(e)
            if not refs:
                continue  # entry has no chunk reference field at all
            entry_valid = all(r in valid_ids for r in refs)
            if entry_valid:
                valid_count += 1
            else:
                invalid_count += 1
                bad_refs = [r for r in refs if r not in valid_ids]
                invalid_samples.append({
                    "query": e.get("query") or e.get("question") or "",
                    "bad_chunk_ids": bad_refs,
                })

        pct_invalid = round(100 * invalid_count / total, 2) if total else 0.0
        print(f"{rel_path:<45} {total:>7} {valid_count:>7} {invalid_count:>8} {pct_invalid:>8}%")

        results[rel_path] = {
            "total": total,
            "valid": valid_count,
            "invalid": invalid_count,
            "pct_invalid": pct_invalid,
            "is_focus_file": rel_path in FOCUS_FILES,
        }

        # collect a few concrete examples where doc still exists under different chunk id
        for sample in invalid_samples[:20]:
            for bad_id in sample["bad_chunk_ids"]:
                doc_guess = extract_doc_id_from_chunk_id(bad_id)
                candidates = doc_to_chunk_ids.get(doc_guess, [])
                if candidates:
                    example_pool.append({
                        "file": rel_path,
                        "query": sample["query"][:80],
                        "old_chunk_id": bad_id,
                        "doc_id_guess": doc_guess,
                        "current_chunk_ids_for_doc": candidates[:5],
                    })
                if len(example_pool) >= 5:
                    break
            if len(example_pool) >= 5:
                break

    print("\n" + "=" * 80)
    print("CONTOH KASUS: doc_id masih ada, chunk_id berubah (rename/rebuild)")
    print("=" * 80)
    if not example_pool:
        print("(Tidak ditemukan contoh otomatis — kemungkinan doc_id encoding berbeda "
              "dari asumsi script, cek manual beberapa invalid id di atas)")
    for ex in example_pool[:5]:
        print(f"\n  file       : {ex['file']}")
        print(f"  query      : {ex['query']}")
        print(f"  old_id     : {ex['old_chunk_id']}")
        print(f"  doc_id     : {ex['doc_id_guess']}")
        print(f"  current_ids: {ex['current_chunk_ids_for_doc']}")

    print("\n" + "=" * 80)
    print("FOKUS: Phase 2 active benchmark files")
    print("=" * 80)
    for f in FOCUS_FILES:
        r = results.get(f, {})
        if "error" in r:
            print(f"  {f}: {r['error']}")
        else:
            flag = "⚠️  SIGNIFICANT" if r.get("pct_invalid", 0) >= 5 else "✅ minor/noise"
            print(f"  {f}: {r.get('invalid',0)}/{r.get('total',0)} invalid "
                  f"({r.get('pct_invalid',0)}%) -> {flag}")

    return {"per_file": results, "examples": example_pool}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argumbent("--base-dir", default=".", help="root project rag-research")
    parser.add_argument("--save", default=None, help="optional path to save JSON summary")
    args = parser.parse_args()

    summary = audit(args.base_dir)

    if args.save and summary:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] {out_path}")