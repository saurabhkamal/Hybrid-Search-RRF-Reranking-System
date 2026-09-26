# reranker.py
# Use an LLM to score how relevant each fused chunk is to the query, then reorders by that score
# pipeline.py calls this last, after fusion.py, right before the final chunks go to the answering LLM

from rag.chat_model import EuriChatModel      # resused here for scoring, and not generation

_SYSTEM_PROMPT = "You are a document re-ranker."

_RERANK_PROMPT = """Evaluate how useful this document is for answering the query.

Query = {query}
Document = {document}

Give a relevance score from 0 to 10:
    10  = directly answers the query
    7-9 = highly relevant
    4-6 = partially relevant
    1-3 = weekly relevant
    0   = completely irrelevant

Return only the numeric score."""
# the actual scoring instruction - {query} and {document} get filled in fresh on every call

# comes after: fusion.py - chunk is one entry from the fused, deduplicated list reciprocol_rank_fusion() returned
def _score_chunk(chat_model: EuriChatModel, query: str, chunk: dict) -> float:
    # asks the LLM for one relevance score for one chunk, and turns its reply into a real number
    prompt = _RERANK_PROMPT.format(query=query, document=chunk["text"])  # fills the template with this call's actual query and chunk text
    reply = chat_model(_SYSTEM_PROMPT, prompt)   # sends the filled prompt to EURI, gets back a plain string reply

    try:
        return float(reply.strip())
    except ValueError:
        return 0.0
        # if the LLM ever replies with something non-numeric, treat it as no signal rather than crashing the whole rerank

# leads into: rerank() below — this is the per-chunk step rerank()'s loop calls once per chunk

# comes after: fusion.py - chunks is the ranked list reciprocal_rank_fusion() already returned
def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    chat_model = EuriChatModel()

    scored = [(chunk, _score_chunk(chat_model, query, chunk)) for chunk in chunks]
    # scores every chunk once, keeping each chunk paired with its own score

    scored.sort(key=lambda pair: pair[1], reverse=True)
    # reorders the pairs by score, highest relevance first

    return [chunk for chunk, score in scored[:top_k]]
    # drops the scores, keeps only the chunks themselves, best top_k first


# leads into: pipeline.py — the list this returns is what actually gets shown to the answering LLM