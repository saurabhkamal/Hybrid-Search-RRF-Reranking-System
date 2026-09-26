# eval/run_comparison.py
# Runs every test query through all four pipeline variants, scores each with metrics.py and prints one comparison table
# This is the file that actually answers "Vector Only vs BM25 Only vs Hybrid vs Hybrid + Reranking"

from eval.test_queries import TEST_QUERIES
from eval.metrics import evaluate, hit_at_k, reciprocal_rank
from rag.pipeline import vector_only, bm25_only, hybrid, hybrid_reranked

VARIANTS = {
    "Vector Only": vector_only,
    "BM25 Only": bm25_only,
    "Hybrid": hybrid,
    "Hybrid + Reranking": hybrid_reranked,
}
# name -> pipeline function, in the exact order the brief lists them, so the table prints in that order

# comes after: pipeline.py — retrieve_fn is one of vector_only/bm25_only/hybrid/hybrid_reranked
def run_variant(name: str, retrieve_fn) -> list[dict]:
    # builds one variant's query results - the single list evaluate() needs
    query_results = []
    total = len(TEST_QUERIES)
    for i, entry in enumerate(TEST_QUERIES, start=1):
        print(f"\r {name}: query {i}/{total}", end="", flush=True)
        # hybrid_reranked alone fires ~10 LLM calls per query, so this can take a while — one line, updated in place
        retrieved = retrieve_fn(entry["query"])
        # runs this one variant's retrieval for this one test query
        query_results.append({
            "query": entry["query"],
            "retrieved": retrieved,
            "relevant_sources": entry["relevant_sources"],
        })
        # bundles the fresh result with the ground truth already sitting in test_queries.py
    print()
    return query_results   # leads into: metrics.evaluate(), called right after this in run() below

# comes after: run_variant() — query_results is exactly what that function just built
def print_per_query(name: str, query_results: list[dict]):
    # shows hit/miss and rank for every individual test query, for one variant
    # reuses hit_at_k and reciprocal_rank directly — no new scoring logic, just finer-grained output
    print(f"\n{name}")
    for i, qr in enumerate(query_results, start=1):
        hit = hit_at_k(qr["retrieved"], qr["relevant_sources"])
        rr = reciprocal_rank(qr["retrieved"], qr["relevant_sources"])
        rank = str(round(1 / rr)) if rr > 0 else "-"
        # rr is 1/rank when there's a hit, so 1/rr recovers the rank itself; rr=0 means no hit in top_k at all
        status = "HIT " if hit else "MISS"
        print(f"  {i:>2}. [{status} rank={rank:<3}] {qr['query'][:65]}")

# comes after: run() below — results is {variant_name: {"hit_rate":..., "mrr":..., "ndcg":...}}
def print_table(results: dict[str, dict]):
    # formats the four variants' scores into one aligned table, printed to the terminal
    header = f"{'Variant':<22}{'Hit Rate':>10}{'MRR':>10}{'NDCG':>10}"
    print(header)
    print("-" * len(header))
    for name, scores in results.items():
        print(f"{name:<22}{scores['hit_rate']:>10.3f}{scores['mrr']:>10.3f}{scores['ndcg']:>10.3f}")

def run():
    # orchestrates the full comparison — every variant, every query, every metric, one table
    all_query_results = {}
    results = {}
    for name, retrieve_fn in VARIANTS.items():
        query_results = run_variant(name, retrieve_fn)
        all_query_results[name] = query_results
        results[name] = evaluate(query_results)

    for name, query_results in all_query_results.items():
        print_per_query(name, query_results)
        # printed for all four variants together, after every variant has finished

    print()
    print_table(results)


if __name__ == "__main__":
    run()
    # so this file can be run directly: python -m eval.run_comparison