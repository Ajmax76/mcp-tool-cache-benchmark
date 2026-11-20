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
