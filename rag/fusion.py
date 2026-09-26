# fusion.py
# Combines the dense (vector) and sparse (BM25) ranked lists into one ranking, using Reciprocol Rank Fusion (RRF)
# pipeline.py calls this after both retrievers have run, and before the reranker.
# comes after: vector_store.query() and sparse_retriever.query() — both ranked lists must already exist before this runs


def reciprocal_rank_fusion(dense_results: list[dict], sparse_results: list[dict], k: int = 60, top_k: int = 10) -> list[dict]:
    # turns two separate ranked lists into one ranking, using each chunk's rank position
    scores = {}        # hold one running RRF score per chunk id
    payloads = {}      # hold one copy of each chunk's id+text/source/page, keyed by chunk id

    for rank, result in enumerate(dense_results, start=1):
        # walks the dense list top to bottom, rank starts at 1 for the first (best) result
        chunk_id = result["id"]       # the shared id this chunk was given at ingestion time
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)    # adds this chunk's dense contribution to its running score — 
                                                                       # a low rank (near 1) adds more than a high rank
        payloads[chunk_id] = result

    for rank, result in enumerate(sparse_results, start=1):
        # walks the sparse list top to bottom, same rank-starts-at-1 rule
        chunk_id = result["id"]        # same shared id — this is what lets a chunk found by both retrievers be recognised as one chunk, not two
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)  # adds this chunk's sparse contribution on top of whatever it already has from the dense pass
                                                                     # - a chunk found by both retrievers ends up with the highest combined score.
        payloads.setdefault(chunk_id, result)

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
    # sorts every chunk id by its combined RRF score, highest first, keeps only the top_k

    return [payloads[cid] for cid in ranked_ids]    # turns the ranked ids back into full chunk data

    


