# MCP & OpenAI `tool_calls` Caching Benchmark (10-Engine Reproduction)

Standard AI gateway caches face a structural mismatch on parameterized agentic and MCP `tool_calls`:

1. **Exact-Match Caches (`SHA-256`):** Drop to **0.0% hit rates** whenever a per-request entity ID (`charge_id`, `pod_id`, `mrn_id`) changes.
2. **Dense Semantic & Lexical Caches:** Built for static informational Q&A rather than parameterized function calls. Embedding models treat alphanumeric IDs as minor surface variation, causing cross-request argument collisions and negation bypasses even above `0.95` similarity.

This repository provides a standalone reproduction script (`reproduce_11_engine_gauntlet.py`) testing **10 caching architectures** (including **5 official HuggingFace embedding models**) across 1,000 dynamic-ID and adversarial tool-call prompts, alongside closed-source reference telemetry for a deterministic slot-isolation pre-filter (**Anaglyph v9.1.6**).

---

## 1. Empirical Neural Cosine Similarity (`all-MiniLM-L6-v2`, 384-d L2-Normalized)

| Prompt A (Cached) | Prompt B (Incoming Request) | Cosine Similarity | Cosine Distance | Behavior at `sim >= 0.92` |
| :--- | :--- | :---: | :---: | :--- |
| `Refund 4,500 cents on charge 'ch_99102000112201' for reason "card declined at checkout"!` | `Refund 4,500 cents on charge 'ch_3MtwBwLkdI9999' for reason "card declined at checkout"!` | **`0.9749`** | `0.0251` | **Stale ID returned** (`ch_99102000112201`) |
| `Refund 4,500 cents on charge 'ch_99102000112201' for reason "card declined at checkout"!` | `Do NOT refund 4,500 cents on charge 'ch_99102000112201' for reason "card declined at checkout"!` | **`0.9896`** | `0.0104` | **Negation bypassed** (cached refund returned) |
| `Scale deployment (dep_pay_0001) to 12 replicas.` | `Scale deployment (dep_pay_0001) to 95 replicas.` | **`0.8946`** | `0.1054` | Collides at `0.85–0.89` thresholds |

---

## 2. Workload Breakdown (`1,000` Calls)

- **450 Valid Dynamic-ID Tool Calls** (unique entity IDs per turn across 4 schemas):
  - `120` `k8s_get_pod_logs` (`pod_id`, `namespace`)
  - `120` `stripe_refund_with_reason` (`amount_cents`, `charge_id`, `reason`)
  - `100` `k8s_scale_deployment` (`deployment`, `replicas` with schema max `100`)
  - `110` `ehr_get_patient_chart` (`mrn_id`, `clinic_code`)
- **550 Adversarial / Must-Bypass Turns:**
  - `150` negation (`"Do NOT refund..."`)
  - `150` out-of-range integer (`replicas: 250` exceeding schema bound `[1, 100]`)
  - `150` multi-turn coreference (`"Do the same for pod_web0001 in staging-eu."`)
  - `100` suffix injection (`'ch_abc_EVIL_SUFFIX'`)

---

## 3. Benchmark Results

Run the 10-engine competitor reproduction locally:

```bash
pip install sentence-transformers scikit-learn numpy
python3 reproduce_11_engine_gauntlet.py
```

| # | Engine / Architecture | Cache Hits | Exact Args | Stale Args Returned | Latency (Cold CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | Exact `SHA-256` (Cloudflare / Kong default) | `0 / 1,000` | `0 (0.0%)` | `0` | `2.2 µs` |
| 2 | HF `all-MiniLM-L6-v2` (LiteLLM / GPTCache, `sim >= 0.92`) | `732 / 1,000` | `0 (0.0%)` | `732` | `14,600.3 µs` |
| 3 | HF `BAAI/bge-small-en-v1.5` (RedisVL, `sim >= 0.92`) | `839 / 1,000` | `0 (0.0%)` | `839` | `27,420.6 µs` |
| 4 | HF `thenlper/gte-small` (Alibaba / Milvus, `sim >= 0.92`) | `839 / 1,000` | `0 (0.0%)` | `839` | `166,089.9 µs` |
| 5 | HF `intfloat/e5-small-v2` (Azure / Elastic, `sim >= 0.92`) | `799 / 1,000` | `0 (0.0%)` | `799` | `26,129.6 µs` |
| 6 | HF `paraphrase-MiniLM-L3-v2` (Fast ONNX, `sim >= 0.92`) | `711 / 1,000` | `0 (0.0%)` | `711` | `7,647.4 µs` |
| 7 | Sklearn TF-IDF Char-Ngram (`sim >= 0.92`) | `0 / 1,000` | `0 (0.0%)` | `0` | `498.9 µs` |
| 8 | IDF-Weighted Token Overlap (`>= 0.85`) | `0 / 1,000` | `0 (0.0%)` | `0` | `333.5 µs` |
| 9 | 4-Shingle Jaccard (`>= 0.85`) | `198 / 1,000` | `0 (0.0%)` | `198` | `831.9 µs` |
| 10 | `SequenceMatcher quick_ratio` (`>= 0.90`) | `591 / 1,000` | `0 (0.0%)` | `591` | `35.9 µs` |
| **Ref** | **Anaglyph v9.1.6** *(closed-source C/WASM reference — not executed by this script)* | **`450 / 1,000*`** | **`450 (100.0%)`** | **`0 (0.000% drift)`** | **`12.8 µs p50 / 18.8 µs p95 / 29.2 µs p99`** |

*Notes:*
- *Semantic and lexical similarity caches are designed for static informational Q&A, making them the wrong architectural primitive for dynamic-argument tool calls rather than a bug in the embedding models themselves.*
- *HuggingFace latencies reflect cold CPU embedding inference (`batch_size=32`). Even on GPU (~1 ms), cosine similarity scores (`0.9749`, `0.9896`, `0.8946`) are unchanged.*
- *`*` Anaglyph v9.1.6 is a closed-source reference binary (`anaglyph_v916_portable.so`, 342 KB stripped `x86-64-v3`, zero GPU or vector DB) and is not bundled in this repository. The `450 / 1,000` hit rate reflects the 4 pre-registered tool schemas with schema bounds enforced (`replicas` in `[1, 100]`); all 550 adversarial, negated, out-of-range, and coreference turns fail open to the upstream LLM.*
- *Latency telemetry across runs on shared `x86-64-v3` CPU: `12.1–12.8 µs` p50 (`16.0–18.8 µs` p95, `24.7–29.2 µs` p99) on the 1,000-call live hook loop; `16.8 µs` p50 (`25.3 µs` p95, `56.9 µs` p99) under an 8-worker concurrent burst across a 12-tool catalog; `17.4 µs` p50 (`32.9 µs` median p99, `31.5–41.2 µs` range) across a 5×20,000-call A/B suite.*

---

## 4. Integration Targets & Evaluation

Anaglyph v9.1.6 is built to run as a zero-dependency Tier-0 pre-filter ahead of existing gateway caches and routers:

- **Native Shared Library (`.so`):** Python C-extension, Go `cgo`, Rust `extern "C"` FFI, LuaJIT FFI
- **Edge / Sandbox Target (`.wasm`):** Cloudflare Workers (`workerd`), Vercel Edge, Envoy WASM filters

For gateway integration testing, custom tool-schema benchmarks, or evaluation access, reach out at **ajmax979361@gmail.com** (`@ajmax76`).
