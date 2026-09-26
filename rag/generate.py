# generate.py
# Turns a user's question into a final answer, using the full hybrid + reranking pipeline for retrieval.
# This is the last stage of the pipeline — the LLM call that actually produces what the user reads.

from rag.chat_model import EuriChatModel
from rag.pipeline import hybrid_reranked

_SYSTEM_PROMPT = "You are a helpful assistant answering questions using only the provided context."
# purpose: sets the LLM's role — answer from context, not from its own general knowledge

_ANSWER_PROMPT = """Answer the question using only the context below. If the context doesn't contain
the answer, say so clearly — don't make anything up. Cite the source and page for each fact you use.

Context:
{context}

Question: {query}

Answer:"""
# purpose: the actual generation instruction — {context} and {query} get filled in fresh on every call

# comes after: pipeline.hybrid_reranked() — chunks is its returned list, already reranked
def _format_context(chunks: list[dict]) -> str:
    # turns the reranked chunk list into one readable block of text for the prompt
    return "\n\n".join(
        f"[Source: {chunk['source']}, page {chunk['page']}]\n{chunk['text']}"
        for chunk in chunks
    )
    # each chunk keeps its source and page attached, so the answer below can cite them

# this is the one function that actually produces a final answer
def answer(query: str, top_k: int = 5) -> str:
    # the full pipeline, start to finish — retrieval, fusion, reranking, then generation
    chunks = hybrid_reranked(query, top_k=top_k)
     # the variant that won the comparison — dense + sparse, fused by RRF, reordered by the LLM reranker

    context = _format_context(chunks)
    prompt = _ANSWER_PROMPT.format(context=context, query=query)

    chat_model = EuriChatModel()
    return chat_model(_SYSTEM_PROMPT, prompt)

if __name__ == "__main__":
    # so this file can be run directly: python -m rag.generate
    query = input("Ask a question: ")
    print()
    print(answer(query))


