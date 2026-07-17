from __future__ import annotations
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from src.controller.agent import Agent

TEST_CASES = [
    {
        "query": "What is the attention mechanism in transformer models?",
        "expect": "ANSWERED",
        "note": "Factual, harusnya ada di corpus (Attention Is All You Need)"
    },
    {
        "query": "How does self-attention differ from cross-attention?",
        "expect": "ANSWERED",
        "note": "Reasoning, harusnya bisa dijawab dari corpus"
    },
    {
        "query": "What is the capital of France?",
        "expect": "ABSTAINED",
        "note": "Out of domain — tidak ada di corpus, harusnya abstain"
    },
    {
        "query": "Who won the FIFA World Cup in 2022?",
        "expect": "ABSTAINED",
        "note": "Out of domain — harusnya abstain"
    },
    {
        "query": "What are the main components of the RAG architecture described in the paper?",
        "expect": "ANSWERED",
        "note": "Harusnya ada di corpus"
    },
]

async def run_tests():
    agent = Agent()
    results = []

    print("\n" + "=" * 65)
    print("  MANUAL AGENT TEST")
    print("=" * 65)

    passed = 0
    failed = 0

    for i, tc in enumerate(TEST_CASES):
        query   = tc["query"]
        expect  = tc["expect"]
        note    = tc["note"]

        print(f"\n[{i+1}/{len(TEST_CASES)}] {query[:60]}")
        print(f"  Note   : {note}")
        print(f"  Expect : {expect}")

        try:
            resp = await agent.run(query)
            actual_status = resp.status.value.upper()
            confidence    = resp.state.confidence_score
            answer        = (resp.answer or "")[:150]
            steps         = resp.state.step_count
            stagnation    = resp.state.stagnation_count
            latency       = resp.latency_ms.get("total", 0)

            if expect == "ANSWERED":
                ok = actual_status == "ANSWERED"
            else:
                ok = actual_status in ("ABSTAINED", "FAILED")

            status_str = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1

            print(f"  Result : {actual_status} [{status_str}]")
            print(f"  Confidence : {confidence:.4f}")
            print(f"  Steps      : {steps} | Stagnation: {stagnation}")
            print(f"  Latency    : {latency}ms")
            if actual_status == "ANSWERED":
                print(f"  Answer : {answer}...")

            if confidence in (0.5, 0.55, 0.6, 0.65):
                print(f"  [WARN] Confidence looks flat ({confidence}) — normalization mungkin belum fix")

            results.append({
                "query":      query,
                "expect":     expect,
                "actual":     actual_status,
                "pass":       ok,
                "confidence": confidence,
                "steps":      steps,
                "stagnation": stagnation,
                "latency_ms": latency,
                "answer":     answer,
            })

        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            failed += 1
            results.append({
                "query":  query,
                "expect": expect,
                "actual": "ERROR",
                "pass":   False,
                "error":  str(e),
            })

    # SUMMARY
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Passed : {passed}/{len(TEST_CASES)}")
    print(f"  Failed : {failed}/{len(TEST_CASES)}")

    confidences = [r["confidence"] for r in results if "confidence" in r]
    if confidences:
        c_min = min(confidences)
        c_max = max(confidences)
        c_range = c_max - c_min
        print(f"\n  Confidence range : {c_min:.4f} – {c_max:.4f} (spread={c_range:.4f})")
        if c_range < 0.15:
            print("  [WARN] Confidence spread < 0.15 — masih terlalu flat, cek normalization")
        else:
            print("  [OK] Confidence spread cukup — engine sudah bervariasi")

    print("\n  Per-query:")
    for r in results:
        flag = "PASS" if r.get("pass") else "FAIL"
        conf = f"{r['confidence']:.4f}" if "confidence" in r else "N/A"
        print(f"    [{flag}] {r['query'][:50]:<50} | conf={conf} | actual={r.get('actual','?')}")

    # Save hasil ke file
    import json
    from pathlib import Path
    out = Path("results/logs/manual_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → {out}")
    print("=" * 65)
    return passed, failed

if __name__ == "__main__":
    asyncio.run(run_tests())