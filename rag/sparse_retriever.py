# sparse_retriever.py
# Handles keyword based search using BM25 - no external database, the index lives on memory and disk
# ingest.py uses this to build the BM25 index next to the vector index
# The retrieve node in the graph uses this to search those chunks at question time.

import os          # Build file paths and create folders
import re          # pull out each word (letters/digits) from a piece of text, ignoring punctuation and spaces
import pickle                      # save the built index to disk and load it back later
from rank_bm25 import BM25Okapi
# scores documents, weighting rare words higher than common ones.

_INDEX_PATH = os.environ.get("BM25_INDEX_PATH", "data/bm25_index.pkl")
# building the BM25 index and using it (searching it) don't have to happen in the same program run.
# the pickle file is what lets "build the index" and "search the index" be two separate, independent runs, instead of one continuous process.

def _tokenize(text: str) -> list[str]:
    # turns one string of text into a list of lowecase words
    return re.findall(r"[a-z0-9]+", text.lower())     # lowercase the text, the pulls out each run of letters / digits as one word


class SparseRetriever:
    # groups the operations ingest.py and the graph nodes need
    
    def __init__(self):
        # runs once the SparseRetriever is created, sets up empty state
        self._bm25 = None     # will hold the built BM25 index
        self._ids = []        # will hold each chunk's id. 
        self._payloads = []   # will hold each chunk's payload


    def build_index(self, ids:list[str], texts:list[str], payloads:list[dict]):
        # builds the BM25 index from the same chunks vector_store.upsert receives
        # texts are the raw chunk strings - BM25 tokenizes directly, it doesn't use embeddings
        tokenized_corpus = [_tokenize(t) for t in texts]    # turns every chunk's text into a list of words, one list per chunk
        self._bm25 = BM25Okapi(tokenized_corpus)            # builds the BM25 index over all the tokenized chunks
        self._ids = ids             # keeps the ids, so a matched chunk can be identified later
        self._payloads = payloads    # keeps the payloads, so a matched chunk's text/source/page can be returned later

        os.makedirs(os.path.dirname(_INDEX_PATH), exist_ok=True)
        # creates the folder for the index file, if it doesn't already exist
        with open(_INDEX_PATH, "wb") as f:
            # opens the index file for writing, in binary mode
            pickle.dump({"bm25": self._bm25, "ids": self._ids, "payloads": self._payloads}, f)
            # saves the index, ids, and payloads together into that one file
            # saves to disk so query() can load it without re-tokenizing every chunk again

    # Brings a BM25 index back into memory that was built and saved earlier
    # comes after build_index() and leads into query()
    def load_index(self):
        with open(_INDEX_PATH, "rb") as f:     # opens the saved index file for reading, in binary mode
            state = pickle.load(f)             # loads the saved dictionary back into memory
        self._bm25 = state["bm25"]             # restores the built BM25 index
        self._ids = state["ids"]               # restores the ids
        self._payloads = state["payloads"]     # restores the payloads

    # query(): this is the actual search step - everything above it just gets an index ready for this moment
    # comes after load_inde() - makes sure an index actually exists in memory before trying to search it.
    # leads into fusion.py - the list this returns gets combined there with vector_store.query()'s list
    def query(self, query_text: str, top_k: int=5) -> list[dict]:
        if self._bm25 is None:              # true if build_index() hasn't run in this process yet
            self.load_index()
            # loads the already-built index from disk instead

        scores = self._bm25.get_scores(_tokenize(query_text))     # scores every chunk in the corpus against the tokenized query
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [{"id": self._ids[i], **self._payloads[i]} for i in ranked_indices]
        # builds one result per top chunk: its id plus its stored text/source/page
        # id is included, matching the shape vector_store.query() now returns too

