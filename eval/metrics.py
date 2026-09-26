# eval/metrics.py
# Scores a pipeline variant's retrieved chunks against the ground truth in test_queries.py
# eval/run_comparison.py calls evaluate() once per variant, across all test queries, to build the comparison table.

import math    # used for log2 inside ndcg

def _is_relevant(result: dict, relevant_set: set) -> bool:
    # one place that decides whether a single retrieved chunk counts as a hit
    return (result.get("source"), result.get("page")) in relevant_set


# ---------------------------------------------------------------------------
# Hit Rate
# ---------------------------------------------------------------------------
# comes after: pipeline.py — retrieved is one variant's output for one query
def hit_at_k(retrieved: list[dict], relevant_sources: list[tuple]) -> int:
    # 1 if any retrieved chunk matches ground truth, 0 otherwise.
    relevant_set = set(relevant_sources)
    return 1 if any(_is_relevant(r, relevant_set) for r in retrieved) else 0
    # any() stops at the first match — hit rate only cares whether at least one exists


def hit_rate(query_results: list[dict]) -> float:
    # fraction of test queries that got at least one relevant chunk
    # hit_at_k - reuses it once per entry in query_results
    hits = [hit_at_k(qr["retrieved"], qr["relevant_sources"]) for qr in query_results]
    return sum(hits)/len(hits)
# leads into: evaluate() below, and the comparison table in run_comparison.py    


# ---------------------------------------------------------------------------
# Mean Reciprocal Rank (MRR)
# ---------------------------------------------------------------------------
def reciprocal_rank(retrieved: list[dict], relevant_sources: list[tuple]) -> float:
    # 1 / rank of the first relevant chunk — for a single query
    relevant_set = set(relevant_sources)
    for i, result in enumerate(retrieved, start=1):
        # i starts at 1, so the first item in the list is rank 1
        if _is_relevant(result, relevant_set):
            return 1 / i
            # stops at the first match — later matches don't matter for MRR
    return 0.0
    # no relevant chunk anywhere in the list

def mrr(query_results: list[dict]) -> float:
    # reciprocal_rank averaged across every test query
    scores = [reciprocal_rank(qr["retrieved"], qr["relevant_sources"]) for qr in query_results]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Normalized Discounted Cumulative Gain (NDCG)
# ---------------------------------------------------------------------------
def _dcg(relevance_scores: list[int]) -> float:
    # sums discounted gain across a list of 0/1 relevance scores, in the order given
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevance_scores))
    # i is 0-indexed, so rank = i + 1, and log2(rank + 1) = log2(i + 2)

def ndcg(retrieved: list[dict], relevant_sources: list[tuple]) -> float:
    # Are relevant chunks placed near the top of the ranking - for a single query
    # ground truth here is binary (a page either answers the query or it doesn't), so relevance is 0 or 1,
    relevant_set = set(relevant_sources)
    relevant_scores = [1 if _is_relevant(r, relevant_set) else 0 for r in retrieved]
    # one 0 or 1 per retrieved chunk, in the order it was actually returned

    ideal_scores = sorted(relevant_scores, reverse=True)
    # the same 1s and 0s, but reordered so every 1 comes first — the best possible ranking

    ideal = _dcg(ideal_scores)
    return _dcg(relevant_scores) / ideal if ideal > 0 else 0.0
# leads into: mean_ndcg 

def mean_ndcg(query_results: list[dict]) -> float:
    # ndcg averaged across every test query
    scores = [ndcg(qr["retrieved"], qr["relevant_sources"]) for qr in query_results]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------
# comes after: pipeline.py - each entry's "retrieved" is one variant's output for that entry's query
def evaluate(query_results: list[dict]) -> dict:
    # runs every metric above for one pipeline variant, across all test queries, in one call
    return {
        "hit_rate": hit_rate(query_results),
        "mrr": mrr(query_results),
        "ndcg": mean_ndcg(query_results),
    }

# leads into: run_comparison.py — this is exactly one row of the four-variant comparison table
