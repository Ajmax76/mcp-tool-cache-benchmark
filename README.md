# MCP & OpenAI `tool_calls` Caching Benchmark: 10-Engine Industry Gauntlet

Standard AI Gateway caches face a fundamental primitive mismatch on parameterized agent `tool_calls`:

1. **Exact-Match Caches (`SHA-256`):** Achieve **0.0% hit rates** whenever per-request entity identifiers (`charge_id`, `pod_id`, `mrn_id`) change.
2. **Semantic & Lexical Similarity Caches:** Built for static informational Q&A, not parameterized tool-call routing. Because dense embedding transformers compress sentence semantics into fixed-width vectors, swapping a customer ID or inserting a negation token (`NOT`) still scores **0.97–0.99 cosine similarity**—replaying Customer A's hardcoded arguments inside Customer B's tool call.

This repository provides a standalone, gateway-agnostic reproduction harness benchmarking **10 standard caching architectures**—5 HuggingFace dense embedding models (`all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `thenlper/gte-small`, `intfloat/e5-small-v2`, `paraphrase-MiniLM-L3-v2`) and 5 exact/lexical/shingle baselines (`SHA-256`, `TF-IDF Char-Ngram`, `IDF-Weighted Token Overlap`, `4-Shingle Jaccard`, `SequenceMatcher`)—against a deterministic slot-isolation pre-filter (**Anaglyph**, zero-dependency C/WASM engine).

---

## 1. Empirical Neural Cosine Similarity (`all-MiniLM-L6-v2`, 384-d L2-Normalized)

| Prompt A (Cached) | Prompt B (Incoming Live Request) | Cosine Similarity | Behavior at threshold `>= 0.92` |
| :--- | :--- | :--- | :--- |
| `Refund 4,500 cents on charge 'ch_99102000112201' for reason "card declined at checkout"!` | `Refund 4,500 cents on charge 'ch_3MtwBwLkdI9999' for reason "card declined at checkout"!` | **`0.9749`** | Returns stale `ch_99102000112201` arguments for Customer B |
| `Refund 4,500 cents on charge 'ch_99102000112201' for reason "card declined at checkout"!` | `Do NOT refund 4,500 cents on charge 'ch_99102000112201' for reason "card declined at checkout"!` | **`0.9896`** | Bypasses negation and returns cached refund tool call |
| `Scale deployment (dep_pay_0001) to 12 replicas.` | `Scale deployment (dep_pay_0001) to 95 replicas.` | **`0.8946`** | Returns stale integer argument (`12` vs `95`) at threshold `<= 0.89` |

---

## 2. Workload Breakdown (1,000 Dynamic-ID & Adversarial Calls)

The deterministic seed (`random.Random(42)`) mixes **450 valid dynamic-ID calls** across 4 tool schemas with **550 adversarial / fail-open calls**:

* **Valid Parameterized Tool Calls (450):**
  * `k8s_get_pod_logs` (120 calls — unique `pod_id`, rotating `namespace`)
  * `stripe_refund_with_reason` (120 calls — unique `charge_id` and `amount_cents`)
  * `k8s_scale_deployment` (100 calls — unique `deployment`, `replicas` in valid schema range `1..100`)
  * `ehr_get_patient_chart` (110 calls — unique `mrn_id`)
* **Adversarial / Must-Bypass Calls (550):**
  * `NEGATION` (150 calls — `"Do NOT refund..."`)
  * `OUT_OF_RANGE` (150 calls — `replicas: 250`, violating tool schema `maximum: 100`)
  * `COREFERENCE` (150 calls — `"Do the same for pod_web0001..."`, ambiguous intent)
  * `SUFFIX_INJECTION` (100 calls — malformed entity ID `'ch_abc_EVIL_SUFFIX'`)

---

## 3. Standalone Reproduction

```bash
git clone [https://github.com/ajmax76/mcp-tool-cache-benchmark.git](https://github.com/ajmax76/mcp-tool-cache-benchmark.git)
cd mcp-tool-cache-benchmark
pip install sentence-transformers scikit-learn numpy
python3 reproduce_11_engine_gauntlet.py
Benchmark Results (1,000 Calls)#Engine / Model ArchitectureCache HitsExact ArgsStale Args ReturnedLatency (Cold CPU)1Exact SHA-256 (exact-match gateway mode)0 / 1,0000 (0.0%)02.2 µs2HF all-MiniLM-L6-v2 (common LiteLLM / GPTCache default)732 / 1,0000 (0.0%)73214,600.3 µs3HF BAAI/bge-small-en-v1.5 (common RedisVL default)839 / 1,0000 (0.0%)83927,420.6 µs4HF thenlper/gte-small839 / 1,0000 (0.0%)839166,089.9 µs5HF intfloat/e5-small-v2799 / 1,0000 (0.0%)79926,129.6 µs6HF paraphrase-MiniLM-L3-v2711 / 1,0000 (0.0%)7117,647.4 µs7Sklearn TF-IDF Char-Ngram (sim >= 0.92)0 / 1,0000 (0.0%)0498.9 µs8IDF-Weighted Token Overlap (BM25-style, >= 0.85)0 / 1,0000 (0.0%)0333.5 µs94-Shingle Jaccard Similarity (Exact LSH upper bound, >= 0.85)198 / 1,0000 (0.0%)198831.9 µs10SequenceMatcher (quick_ratio >= 0.90)591 / 1,0000 (0.0%)59135.9 µsRefAnaglyph v9.1.6 (closed-source C/WASM reference — not executed by script)450 / 1,000*450 (100.0%)0 (0.000% drift)12.8 µs p50 / 17.4 µs mean (p99=32.9 µs)*Notes:HuggingFace transformer latencies reflect cold CPU embedding inference (batch_size=32). Even on pre-warmed GPU (~1 ms), cosine similarity scores (0.9749 and 0.9896) are invariant.Anaglyph's 450 / 1,000 result reflects pre-registered tool schemas for the 4 valid tools (on a cold, zero-schema warm-up, the first call per template misses for 446 / 450 hits) and fails open cleanly on all 550 adversarial/out-of-range calls.
