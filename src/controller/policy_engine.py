"""
src/controller/policy_engine.py

PolicyEngine — centralized decision layer for RAG pipeline.

Kenapa ini ada:
  Sebelumnya semua keputusan (query type, retrieval strategy, reference policy,
  generation config) tersebar di run_single_query.py dan generator.py.
  Efeknya: sulit di-audit, sulit dikontrol, sulit di-test.

  PolicyEngine mengkonsolidasi semua decision ke satu tempat.
  Pipeline tidak lagi "tahu" bagaimana membuat keputusan —
  pipeline hanya "tahu" bagaimana mengeksekusi keputusan.

  Separation: PolicyEngine = decision  |  Pipeline = execution
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Type alias — supaya return type jelas di seluruh codebase
# ---------------------------------------------------------------------------
QueryType = str   # "factual" | "reasoning" | "general"


@dataclass
class PlanHints:
    query_type: QueryType
    preferred_strategy: str
    expected_complexity: str
    allow_multi_query: bool
    allow_hyde: bool
    allow_references: bool
    max_ref_ratio: float
    top_k_hint: int
    generation_hint: dict[str, float]


class PolicyEngine:
    """
    Single source of truth untuk semua keputusan pipeline.

    Semua method bersifat pure function:
      - Input deterministik → output deterministik
      - Tidak ada side effect
      - Tidak ada state yang berubah antar call
      - Mudah di-unit-test tanpa mock

    Usage:
        policy = PolicyEngine()
        q_type = policy.decide_query_type(query)
        ret    = policy.retrieval_strategy(q_type)
        ref    = policy.reference_policy(query, q_type)
        gen    = policy.generation_policy(q_type)
    """

    # -----------------------------------------------------------------------
    # 1. Query Type Classification
    # -----------------------------------------------------------------------

    def decide_query_type(self, query: str) -> QueryType:
        """
        Klasifikasi query ke satu dari tiga tipe.

        Rule order penting:
          "who wrote how X works" → "who" ditemukan duluan → factual
          Ini intentional: factual lebih strict, lebih aman, lebih murah.

        Edge cases yang di-handle:
          - Query kosong → "general" (safe default)
          - Query dengan tanda tanya / huruf besar → strip + lower dulu
          - Multi-signal query (ada "how" dan "who") → first-match wins (factual > reasoning > general)

        Returns:
            "factual"   — query meminta satu fakta spesifik (siapa, kapan, di mana)
            "reasoning" — query meminta penjelasan atau alasan (bagaimana, kenapa)
            "general"   — query lainnya, biasanya definitional atau komparatif
        """
        if not query or not query.strip():
            return "general"

        q = query.lower().strip()

        # Factual: single-fact lookup — deterministic, tidak butuh kreativitas LLM
        FACTUAL_SIGNALS = {"who", "when", "where", "which", "whom"}
        if any(word in q.split() or q.startswith(word) for word in FACTUAL_SIGNALS):
            return "factual"

        # Reasoning: membutuhkan multi-step inference — butuh lebih banyak context
        REASONING_SIGNALS = {"how", "why", "explain", "describe", "what causes", "what makes"}
        if any(signal in q for signal in REASONING_SIGNALS):
            return "reasoning"

        # General: definisi, komparasi, atau apapun yang tidak masuk dua kategori di atas
        return "general"

    # -----------------------------------------------------------------------
    # 2. Retrieval Strategy
    # -----------------------------------------------------------------------

    def retrieval_strategy(self, query_type: QueryType) -> dict:
        """
        Tentukan strategi retrieval berdasarkan query type.

        Kenapa factual tidak pakai HyDE:
          HyDE menghasilkan dokumen hipotetis untuk memperluas coverage.
          Untuk factual query ("who wrote X?"), HyDE sering menghasilkan
          dokumen yang justru drift dari topik — menambah noise, bukan signal.

        Kenapa reasoning pakai HyDE + multi_query:
          Reasoning query butuh perspektif lebih luas.
          HyDE membantu jika jawaban tidak tersedia verbatim di corpus.
          Multi-query meng-cover berbagai angle pertanyaan yang sama.

        Kenapa general tidak pakai HyDE:
          HyDE mahal (satu LLM call extra).
          Untuk definisi / komparasi, multi_query sudah cukup untuk coverage.

        Returns dict dengan key:
            use_hyde        (bool)  — generate hypothetical document sebelum retrieval
            use_multi_query (bool)  — expand query jadi beberapa variasi
            top_k           (int)   — jumlah chunks yang di-retrieve
        """
        _STRATEGIES: dict[QueryType, dict] = {
            "factual": {
                "use_hyde":        False,
                "use_multi_query": False,
                "top_k":           5,
                # Factual tidak perlu ekspansi — query sudah spesifik.
                # Ekspansi justru menambah noise untuk lookup sederhana.
            },
            "general": {
                "use_hyde":        False,
                "use_multi_query": True,
                "top_k":           5,
                # General butuh coverage lebih luas → multi_query ON.
                # HyDE tidak perlu karena query definitional sudah cukup eksplisit.
            },
            "reasoning": {
                "use_hyde":        True,
                "use_multi_query": True,
                "top_k":           5,
                # Reasoning butuh maksimal coverage dan depth.
                # HyDE + multi_query = dua mekanisme ekspansi berbeda yang saling melengkapi.
            },
        }

        strategy = _STRATEGIES.get(query_type)
        if strategy is None:
            # Unknown query_type → safe default (sama dengan "general")
            return {"use_hyde": False, "use_multi_query": True, "top_k": 5}

        return strategy

    # -----------------------------------------------------------------------
    # 3. Reference Policy
    # -----------------------------------------------------------------------

    def reference_policy(self, query: str, query_type: QueryType) -> dict:
        """
        Kontrol apakah reference/bibliography chunks boleh masuk ke context.

        Ini adalah titik kontrol untuk "reference leakage" — masalah nyata
        yang terjadi di pipeline sebelumnya (ref_ratio mencapai 0.6–0.8).

        Kenapa factual BOLEH reference:
          Pertanyaan "who are the authors?" jawabannya ada di reference section.
          Hard-block reference untuk factual → INSUFFICIENT_CONTEXT untuk kasus ini.
          Soft limit (max_ref_ratio=0.4) tetap mencegah reference mendominasi.

        Kenapa reasoning TIDAK BOLEH reference:
          Reasoning query butuh konten substantif (body, abstract, conclusion).
          Reference section berisi metadata (nama, tahun, DOI) — bukan penjelasan.
          Membiarkan reference masuk untuk reasoning → model "menjelaskan" dari citation list
          → jawaban dangkal, sering hallucinate isi paper dari judulnya saja.

        Kenapa general TIDAK BOLEH reference:
          Sama seperti reasoning — general query (definisi, komparasi) butuh konten,
          bukan metadata bibliografi.

        Signal override:
          Selain query_type, kita juga cek kata kunci eksplisit di query text.
          "who wrote", "author", "citation" → override ke allow regardless of type.
          Ini menangkap edge case di mana query_type salah klasifikasi.

        Returns dict dengan key:
            allow_references (bool)  — apakah reference chunks boleh masuk context
            max_ref_ratio    (float) — max proporsi reference di final context (0.0–1.0)
        """
        # Signal override: explicit citation keywords → always allow
        # Ini safety net kalau query_type salah klasifikasi
        CITATION_SIGNALS = {
            "author", "authors", "who wrote", "published by",
            "citation", "cite", "reference", "bibliography",
            "journal", "proceedings", "paper by",
        }
        q_lower = query.lower()
        has_citation_signal = any(signal in q_lower for signal in CITATION_SIGNALS)

        if has_citation_signal:
            return {
                "allow_references": True,
                "max_ref_ratio":    0.4,
                # Citation query → reference boleh, tapi tetap dibatasi 40%
                # supaya non-ref chunks masih punya ruang
            }

        _POLICIES: dict[QueryType, dict] = {
            "factual": {
                "allow_references": True,
                "max_ref_ratio":    0.4,
                # Factual bisa butuh reference untuk author/dataset lookup
            },
            "reasoning": {
                "allow_references": False,
                "max_ref_ratio":    0.0,
                # Reasoning butuh konten — reference diblok keras
            },
            "general": {
                "allow_references": False,
                "max_ref_ratio":    0.0,
                # General butuh konten — reference diblok keras
            },
        }

        policy = _POLICIES.get(query_type)
        if policy is None:
            # Unknown query_type → safe default: blok reference
            return {"allow_references": False, "max_ref_ratio": 0.0}

        return policy

    # -----------------------------------------------------------------------
    # 4. Generation Policy
    # -----------------------------------------------------------------------

    def generation_policy(self, query_type: QueryType) -> dict:
        """
        Konfigurasi LLM generation berdasarkan query type.

        Kenapa factual pakai temperature=0.0:
          Factual query ("when was X published?") punya satu jawaban benar.
          Temperature > 0 memperkenalkan variasi yang tidak diinginkan.
          Deterministic output = mudah di-test, mudah di-compare antar run.

        Kenapa reasoning pakai max_tokens lebih tinggi:
          Reasoning query membutuhkan chain of thought.
          Memotong jawaban di 100 tokens untuk reasoning = truncasi di tengah argumen.
          300 tokens memberi ruang untuk multi-step explanation.

        Kenapa general di tengah:
          Definisi dan komparasi butuh lebih dari 100 token (terlalu pendek untuk definisi)
          tapi tidak perlu 300 (tidak butuh reasoning chain panjang).
          200 token + sedikit kreativitas (0.2) untuk variasi paraphrase.

        Returns dict dengan key:
            max_tokens  (int)   — batas token output LLM
            temperature (float) — kreativitas LLM (0.0 = deterministic, 1.0 = creative)
        """
        _POLICIES: dict[QueryType, dict] = {
            "factual": {
                "max_tokens":  100,
                "temperature": 0.0,
            },
            "general": {
                "max_tokens":  200,
                "temperature": 0.2,
            },
            "reasoning": {
                "max_tokens":  300,
                "temperature": 0.3,
            },
        }

        policy = _POLICIES.get(query_type)
        if policy is None:
            # Unknown query_type → safe default (sama dengan "general")
            return {"max_tokens": 200, "temperature": 0.2}

        return policy

    # -----------------------------------------------------------------------
    # Convenience: get all policies in one call
    # -----------------------------------------------------------------------

    def resolve(self, query: str) -> dict:
        """
        Satu call untuk semua keputusan — cocok untuk pipeline yang butuh semua sekaligus.

        Usage:
            decision = policy.resolve(query)
            q_type   = decision["query_type"]
            ret      = decision["retrieval"]
            ref      = decision["reference"]
            gen      = decision["generation"]

        Returns nested dict yang bisa langsung di-unpack ke pipeline.
        """
        q_type = self.decide_query_type(query)
        return {
            "query_type": q_type,
            "retrieval":  self.retrieval_strategy(q_type),
            "reference":  self.reference_policy(query, q_type),
            "generation": self.generation_policy(q_type),
        }

    def initial_plan(self, query: str) -> PlanHints:
        q_type = self.decide_query_type(query)
        retrieval = self.retrieval_strategy(q_type)
        reference = self.reference_policy(query, q_type)
        generation = self.generation_policy(q_type)

        preferred_strategy = "hybrid" if q_type != "factual" else "dense"
        expected_complexity = (
            "high" if q_type == "reasoning" else "medium" if q_type == "general" else "low"
        )

        return PlanHints(
            query_type=q_type,
            preferred_strategy=preferred_strategy,
            expected_complexity=expected_complexity,
            allow_multi_query=retrieval["use_multi_query"],
            allow_hyde=retrieval["use_hyde"],
            allow_references=reference["allow_references"],
            max_ref_ratio=reference["max_ref_ratio"],
            top_k_hint=retrieval["top_k"],
            generation_hint=generation,
        )
