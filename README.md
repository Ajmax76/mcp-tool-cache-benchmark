# MCP & OpenAI `tool_calls` Caching Benchmark: 11-Engine Industry Gauntlet

Standard AI Gateway caches face a fundamental primitive mismatch on parameterized agent `tool_calls`:
1. **Exact-Match Caches (`SHA-256`):** Achieve **0.0% hit rates** whenever entity identifiers (`charge_id`, `pod_id`, `mrn_id`) change per ticket.
2. **Semantic & Lexical Similarity Caches:** Are designed for static informational Q&A, making them the wrong architectural primitive for parameterized tool-call traffic with dynamic per-request entity IDs. Because dense embedding transformers compress sentence semantics into fixed-width vectors, changing a customer ID or adding a negation token (`NOT`) produces `0.97–0.99` cosine similarity—returning cached stale arguments from a previous customer's tool call.

This repository contains a standalone reproduction benchmark testing **10 production caching architectures**—including **5 official HuggingFace dense embedding transformers** (`all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `thenlper/gte-small`, `intfloat/e5-small-v2`, `paraphrase-MiniLM-L3-v2`) and **5 sparse/lexical/LSH engines** (`SHA-256 Exact`, `TF-IDF Char-Ngram`, `Okapi BM25`, `MinHash LSH`, `GPTCache SequenceMatcher`)—against **Anaglyph `v9.1.6`**, a compiled `x86-64-v3` binary sidecar for deterministic `tool_calls` synthesis.

---

## 1. Empirical Neural Cosine Similarity (`all-MiniLM-L6-v2`, 384-d L2-Normalized)

| Prompt A (Cached) | Prompt B (Incoming Live Request) | Cosine Similarity | Behavior at `threshold >= 0.92` |
| :--- | :--- | :---: | :--- |
| `Refund 4,500 cents on charge 'ch_99102000112201' for reason "card declined at checkout"!` | `Refund 4,500 cents on charge 'ch_3MtwBwLkdI9999' for reason "card declined at checkout"!` | **`0.9749`** | Returns stale `ch_99102000112201` arguments for Customer B |
| `Refund 4,500 cents on charge 'ch_99102000112201' for reason "card declined at checkout"!` | `Do NOT refund 4,500 cents on charge 'ch_99102000112201' for reason "card declined at checkout"!` | **`0.9896`** | Bypasses negation and returns cached refund call |
| `Scale deployment (dep_pay_0001) to 12 replicas.` | `Scale deployment (dep_pay_0001) to 95 replicas.` | **`0.8946`** | Returns stale integer argument (`12` vs `95`) at `<= 0.89` |

---

## 2. The 11-Engine Benchmark Receipt (1,000 Dynamic-ID & Adversarial Tool Calls)

Run the standalone competitor reproduction locally:
bash
pip install sentence-transformers scikit-learn numpy
python3 reproduce_11_engine_gauntlet.py



================================================================================================================
[11-ENGINE BENCHMARK: 10 PRODUCTION CACHES vs. ANAGLYPH v9.1.6 (x86-64-v3 .so BINARY)]
================================================================================================================
#  Engine / Model Architecture               | Cache Hits | Exact Args    | Stale Args Returned        | Latency (Cold CPU)
------------------------------------------------------------------------------------------------------------
1. Cloudflare / Kong SHA-256 Exact           |    0/1,000 |    0 (  0.0%) |    0 stale args returned   |      2.2 us
2. HF all-MiniLM-L6-v2 (LiteLLM/GPTCache)  |  732/1,000 |    0 (  0.0%) |  732 stale args returned   |  14600.3 us
3. HF BAAI/bge-small-en-v1.5 (RedisVL)     |  839/1,000 |    0 (  0.0%) |  839 stale args returned   |  27420.6 us
4. HF thenlper/gte-small (Alibaba/Milvus)  |  839/1,000 |    0 (  0.0%) |  839 stale args returned   | 166089.9 us
5. HF intfloat/e5-small-v2 (Azure/Elastic) |  799/1,000 |    0 (  0.0%) |  799 stale args returned   |  26129.6 us
6. HF paraphrase-MiniLM-L3-v2 (Fast ONNX)  |  711/1,000 |    0 (  0.0%) |  711 stale args returned   |   7647.4 us
7. Sklearn TF-IDF Char-Ngram (sim>=0.92)     |    0/1,000 |    0 (  0.0%) |    0 stale args returned   |    498.9 us
8. Okapi BM25 Sparse Lexical Cache           |    0/1,000 |    0 (  0.0%) |    0 stale args returned   |    333.5 us
9. MinHash 4-Shingle Jaccard LSH             |  198/1,000 |    0 (  0.0%) |  198 stale args returned   |    831.9 us
10. GPTCache SequenceMatcher (ratio>=0.90)   |  591/1,000 |    0 (  0.0%) |  591 stale args returned   |     35.9 us
------------------------------------------------------------------------------------------------------------
11. Anaglyph v9.1.6 Compiled Binary (.so)    |  450/1,000 |  450 (100.0%) |    0 (0.000% drift)        |     17.4 us (p99=32.9us)
================================================================================================================

*Note: HuggingFace transformer latencies reflect cold CPU embedding inference (`batch_size=32`). Even on pre-warmed GPU (`~1 ms`), cosine similarity values (`0.9749` and `0.9896`) are invariant.
