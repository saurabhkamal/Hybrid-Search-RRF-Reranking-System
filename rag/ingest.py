# ingest.py
# Reads PDF files from data/, splits them into overlapping chunks.
# embeds each chunk, and saves everything in Qdrant.
# Run this once, or whenever the corpus changes, before running the graph.

import os      # used to list files in the data folder
import uuid    # generates valid Qdrant point ids for each chunk
import pdfplumber   # extracts text from PDF pages, cleaner than pypdf for this corpus
from rag.embedding import EuriEmbedder    # turns chunk text into vectors
from rag.vector_store import VectorStore   # saves vectors and payloads into Qdrant
from rag.sparse_retriever import SparseRetriever    # builds the BM25 keyword index

DATA_DIR = "data"     # folder holding the source PDFs
CHUNK_SIZE = 1500     # max characters per chunk, matches what worked best in my earlier project
CHUNK_OVERLAP = 200   # characters repeated between consecutive chunks, so a cut mid-sentence isn't lost

def chunk_text(text: str) -> list[str]:
    # splits one page's text into overlapping pieces of CHUNK_SIZE characters
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        # take the next CHUNK_SIZE characters starting from `start`
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
        # move forward, but step back by CHUNK_OVERLAP so chunks overlap
    return chunks

def load_chunks() -> list[dict]:
    # reads every PDF in DATA_DIR and returns one dict per chunk: text, source, page
    records = []
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".pdf"):
            continue
            # skip anything that isn't a PDF
        path = os.path.join(DATA_DIR, filename)
        with pdfplumber.open(path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if not page_text:
                    continue
                    # skip pages with no extractable text, e.g. scanned images
                for chunk in chunk_text(page_text):
                    records.append({"text": chunk, "source": filename, "page": page_number})

    return records


def print_embedding_progress(batch_num: int, total_batches: int):
    # print the single updating line with percentage, instead of one line per batch
    percent = int(batch_num/total_batches * 100)
    print(f"\rEmbedding chunks: {percent}% ({batch_num}/{total_batches} batches)", end="", flush=True)

def print_upsert_progress(batch_num: int, total_batches: int):
    percent = int(batch_num / total_batches * 100)
    print(f"\rSaving to Qdrant: {percent}% ({batch_num}/{total_batches} batches)", end="", flush=True)

def run():
    # orchestrates the full ingestion: load pages, embed chunks, save to Qdrant
    records = load_chunks()
    print(f"loaded {len(records)} chunks from {DATA_DIR}")
    
    texts = [r["text"] for r in records]
    # pulls out just the text, since that's all the embedder and BM25 need

    embedder = EuriEmbedder()
    vectors = embedder(texts, on_progress=print_embedding_progress)
    print()
    # embeds all chunks in batches of up to 100, printing progress as it goes

    store = VectorStore()
    store.ensure_collection(vector_size=len(vectors[0]))
    # Creates the Qdrant collection now that we know the real vector size

    ids = [str(uuid.uuid4()) for _ in records]
    # one fresh UUID per chunk, reused below for BM25 so a chunk has one id on both sides

    payloads = [{"text": r["text"], "source": r["source"], "page": r["page"]} for r in records]
    # what gets returned later when this chunk is retrieved.

    store.upsert(ids=ids, vectors=vectors, payloads=payloads, on_progress=print_upsert_progress)
    print()
    print(f"ingested {len(records)} chunks from {DATA_DIR}")

    sparse = SparseRetriever()
    sparse.build_index(ids=ids, texts=texts, payloads=payloads)
    # same ids and payloads as Qdrant, so fusion.py can match a chunk across both retrievers
    print(f"built BM25 index over {len(records)} chunks")


if __name__ == "__main__":
    run()
    # so this file can be run directly: python -m rag.ingest


