import sys
sys.path.insert(0, ".")
from src.controller.confidence_engine import ConfidenceEngine

ce = ConfidenceEngine()

def make_chunk(score, doc_id="doc_A"):
    return {"retrieval_score": score, "metadata": {"doc_id": doc_id}}

passed = 0
failed = 0

def check(name, condition, info=""):
    global passed, failed
    if condition:
        print(f"  PASS — {name}")
        passed += 1
    else:
        print(f"  FAIL — {name} {info}")
        failed += 1

print("\n=== Edge Cases ===")
r = ce.calculate_confidence([])
check("T1 empty chunks → REJECT", r["decision"] == "REJECT" and r["confidence_score"] == 0.0)

r = ce.calculate_confidence([{"no_score": True}])
check("T2 no valid scores", r["signals"].get("reason") == "no_valid_scores")

print("\n=== Bug Fix Validasi ===")
chunks_8 = [make_chunk(s) for s in [0.9, 0.85, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1]]
chunks_3 = [make_chunk(s) for s in [0.9, 0.85, 0.8]]
r8 = ce.calculate_confidence(chunks_8)
r3 = ce.calculate_confidence(chunks_3)
diff = abs(r8["confidence_score"] - r3["confidence_score"])
check("T3 chunk count stability (diff < 0.05)",
      diff < 0.05,
      f"diff={diff:.4f} | 8chunks={r8['confidence_score']} | 3chunks={r3['confidence_score']}")

# T4: spurious match
chunks = [make_chunk(0.95)] + [make_chunk(0.1) for _ in range(5)]
r = ce.calculate_confidence(chunks)
check("T4 spurious_match=True", r["signals"]["spurious_match"] == True, str(r["signals"]))
check("T4 spurious → tidak GENERATE", r["decision"] != "GENERATE", f"decision={r['decision']}")

print("\n=== Gap Modifier ===")

# T5: gap besar + mean tinggi → +0.10
chunks = [make_chunk(0.9, "doc_A"), make_chunk(0.3, "doc_B"), make_chunk(0.28, "doc_C")]
r = ce.calculate_confidence(chunks)
print(f"       gap={r['signals']['gap']:.4f} | mean_top3={r['signals']['mean_top3']:.4f} | score={r['confidence_score']} | decision={r['decision']}")
check("T5 gap positive → GENERATE", r["decision"] == "GENERATE", f"decision={r['decision']}")

# T6: gap besar + mean rendah → -0.15
chunks_penalty = [make_chunk(0.9, "doc_A"), make_chunk(0.15, "doc_B"), make_chunk(0.1, "doc_C")]
chunks_no_gap  = [make_chunk(0.5, "doc_A"), make_chunk(0.48, "doc_B"), make_chunk(0.46, "doc_C")]
rp = ce.calculate_confidence(chunks_penalty)
rn = ce.calculate_confidence(chunks_no_gap)
print(f"       penalty_score={rp['confidence_score']} | no_gap_score={rn['confidence_score']}")
check("T6 gap penalty applied (score turun vs baseline)",
      rp["confidence_score"] < rn["confidence_score"],
      f"penalty={rp['confidence_score']} baseline={rn['confidence_score']}")

print("\n=== Source Agreement ===")

# T7: 3 unique docs → agreement = 1.0
chunks = [
    make_chunk(0.8, "doc_A"), make_chunk(0.75, "doc_B"),
    make_chunk(0.7, "doc_C"), make_chunk(0.65, "doc_A"), make_chunk(0.6, "doc_B"),
]
r = ce.calculate_confidence(chunks)
check("T7 agreement=1.0 (3 unique docs)", r["signals"]["agreement"] == 1.0, f"agreement={r['signals']['agreement']}")

# T7b: 1 unique doc → agreement rendah
chunks_single = [make_chunk(s, "doc_A") for s in [0.8, 0.75, 0.7, 0.65, 0.6]]
r = ce.calculate_confidence(chunks_single)
check("T7b agreement<0.5 (1 unique doc)", r["signals"]["agreement"] < 0.5, f"agreement={r['signals']['agreement']}")

print("\n=== Happy Path ===")

chunks = [make_chunk(s, f"doc_{i}") for i, s in enumerate([0.9, 0.85, 0.8])]
r = ce.calculate_confidence(chunks)
check("T8 happy path → GENERATE", r["decision"] == "GENERATE", f"decision={r['decision']}")
check("T8 score > 0.45", r["confidence_score"] > 0.45, f"score={r['confidence_score']}")

print(f"\n{'='*35}")
print(f"  {passed} passed | {failed} failed")
print(f"{'='*35}")