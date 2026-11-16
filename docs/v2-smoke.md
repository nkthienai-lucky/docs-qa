# V2 retrieval smoke

Run in WSL Ubuntu against the indexed English Vocabulary in Use baseline (`3a357be041df101c`, 203 chunks), with the reranker loaded on CPU and Ollama embeddings on the configured local backend.

The smoke exercised `v2a` (dense candidates + cross-encoder), `v2b` (dense/BM25 RRF + cross-encoder), and `v2b_norerank` (dense/BM25 RRF only) on two representative questions:

| Question | Observation |
|---|---|
| What is the difference between borrow and lend? | All configurations returned valid baseline chunks; v2a/v2b top chunk was `:47` (reranker score 0.55450); v2b_norerank ranked `:47` first (RRF 0.03279). The smoke output did not demonstrate the target concept in its excerpts, so this is a pipeline smoke, not a relevance judgment. |
| How do you use present perfect? | All configurations returned valid baseline chunks; v2a/v2b top chunk was `:141` (reranker score 0.36511), while v2b_norerank top chunk was `:91` (RRF 0.03150). The returned excerpts are imperfect matches, so the smoke is not evidence of retrieval quality. |

The run verifies that all three V2 retriever paths execute end-to-end against real pgvector, Ollama embeddings, BM25 over all 203 stored chunks, and the CPU cross-encoder. Quality measurement belongs to the frozen evaluation task.
