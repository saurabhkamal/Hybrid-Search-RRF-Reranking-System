# vector_store.py
# Handles all talking to Qdrant, the vector database.
# ingest.py uses this to save embedded chunks.
# The retrieve node in the graph uses this to search those chunks at question time.

import os                                   
from qdrant_client import QdrantClient      # library for connecting to Qdrant
from qdrant_client.models import PointStruct, VectorParams, Distance
# Distance: which similarity measure to use when comparing vectors
# VectorParams: settings for the collection, like vector size
# PointStruct: one saved item, made of an id, a vector, and a payload

from dotenv import load_dotenv     # loads values from the .env file into the environment

load_dotenv()     # runs the loading, so os.environ can now see QDRANT_URL and QDRANT_API_KEY

_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=60,
)

UPSERT_BATCH_LIMIT = 50

_COLLECTION = os.environ["QDRANT_COLLECTION"]
# name of the Qdrant collection we read from and write to, like a table name

class VectorStore:
    # groups the three operations ingest.py and the graph nodes need

    def ensure_collection(self, vector_size: int):
        # call this once before the first upsert, to create the collection if it doesn't exist yet
        if not _client.collection_exists(_COLLECTION):
            # only create it if missing, so this is safe to call every time the app starts
            _client.create_collection(
                collection_name=_COLLECTION,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                # vector_size must match the embedding model's output length exactly
                # cosine distance is the standard similarity measure for text embeddings
            )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict], on_progress=None):
        # saves chunks into Qdrant: one id, one vector, one payload per chunk
        total = len(ids)
        total_batches = (total + UPSERT_BATCH_LIMIT - 1) // UPSERT_BATCH_LIMIT

        for batch_num, start in enumerate(range(0, total, UPSERT_BATCH_LIMIT), start=1):
            end = start + UPSERT_BATCH_LIMIT
            points = [
                PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
                for i in range(start, min(end, total))
            ]
            # builds only this batch's points, not all of them at once

            _client.upsert(collection_name=_COLLECTION, points=points)

            if on_progress:
                on_progress(batch_num, total_batches)

    
    def query(self, vector: list[float], top_k: int=5) -> list[dict]:
        # find the top_k chunks in Qdrant most similar to the given vector
        results = _client.query_points(collection_name=_COLLECTION, query=vector, limit=top_k)
        # each result ("hit") carries a score, an id, and the payload we stored
        return [{"id": point.id, **point.payload} for point in results.points]
        # id is included so fusion.py can match this chunk against the sparse results
                  