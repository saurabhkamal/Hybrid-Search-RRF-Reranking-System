# pipeline.py
# Wires the dense retriever, sparse retriever, fusion, and reranker into the four variants being compared.
# eval/run_comparison.py calls these four functions, one per test query, to score each retrieval strategy.

from rag.embedding import EuriEmbedder
from rag.vector_store import VectorStore
from rag.sparse_retriever import SparseRetriever
from rag.fusion import reciprocal_rank_fusion
from rag.reranker import rerank

_embedder = EuriEmbedder()
_vector_store = VectorStore()
_sparse_retriever = SparseRetriever()
# one shared instance of each, reused across every call — same pattern as reranker.py's chat_model reuse


def vector_only(query: str, top_k: int = 5) -> list[dict]:
    # retrieval using dense/vector search alone
    query_vector = _embedder([query])[0]    # embeds thr query text into the same vector space the chunks embedded into
    return _vector_store.query(query_vector, top_k=top_k)

def bm25_only(query: str, top_k: int = 5) -> list[dict]:
    # retrieval using BM25/keyword search alone, the second baseline, alongside vector_only
    return _sparse_retriever.query(query, top_k=top_k)

def hybrid(query: str, top_k: int = 5, candidate_k: int = 10) -> list[dict]:
    # retrieval using both dense and sparse search, merged by RRF, no reranking yet
    query_vector = _embedder([query])[0]
    dense_results = _vector_store.query(query_vector, top_k=candidate_k)
    sparse_results = _sparse_retriever.query(query, top_k=candidate_k)
    # candidate_k, not top_k — each retriever contributes a wider pool, so RRF has real overlap to find before narrowing down
    return reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)

def hybrid_reranked(query: str, top_k: int = 5, candidate_k: int = 10) -> list[dict]:
    # the full pipeline — hybrid's fused shortlist, reordered by the LLM reranker
    fused_results = hybrid(query, top_k=candidate_k, candidate_k=candidate_k)  
    # asks hybrid for candidate_k results, not top_k — the reranker needs a real shortlist to narrow down, not an already-final 5
    return rerank(query, fused_results, top_k=top_k)

# leads into: eval/metrics.py and later the answering LLM

