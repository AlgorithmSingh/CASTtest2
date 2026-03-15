"""Simple RAG pipeline using TF-IDF embeddings and cosine similarity.

This provides a lightweight, self-contained retrieval system for comparing
chunking strategies without requiring external embedding models or GPUs.
"""

from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.cast_chunker import Chunk


class SimpleRetriever:
    """TF-IDF-based code retriever for RAG experiments."""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._vectorizer = TfidfVectorizer(
            token_pattern=r"[a-zA-Z_]\w*",  # match identifiers
            lowercase=True,
            max_features=5000,
        )
        contents = [c.content for c in chunks]
        self._tfidf_matrix = self._vectorizer.fit_transform(contents)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Retrieve the top-k most relevant chunks for a query.

        Args:
            query: Natural language or code query.
            top_k: Number of chunks to return.

        Returns:
            List of (chunk, similarity_score) tuples, sorted by relevance.
        """
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]


def evaluate_retrieval(
    retriever: SimpleRetriever,
    queries: List[dict],
    top_k: int = 5,
) -> dict:
    """Evaluate retrieval quality on a set of queries with ground truth.

    Each query dict should have:
        - "query": str — the search query
        - "expected_lines": list of (start_line, end_line) ranges that
          should appear in retrieved chunks
        - "filepath": str — which file the relevant code is in

    Returns:
        Dictionary with precision@k, recall@k, and per-query details.
    """
    total_precision = 0.0
    total_recall = 0.0
    details = []

    for q in queries:
        results = retriever.retrieve(q["query"], top_k=top_k)

        # Check which expected line ranges are covered by retrieved chunks
        expected_ranges = q["expected_lines"]
        retrieved_lines = set()
        for chunk, score in results:
            if chunk.filepath == q["filepath"]:
                for line in range(chunk.start_line, chunk.end_line + 1):
                    retrieved_lines.add(line)

        # Compute how many expected ranges are "hit" (≥50% overlap)
        hits = 0
        for start, end in expected_ranges:
            expected_set = set(range(start, end + 1))
            overlap = len(expected_set & retrieved_lines)
            if overlap >= len(expected_set) * 0.5:
                hits += 1

        recall = hits / len(expected_ranges) if expected_ranges else 0.0

        # Precision: fraction of retrieved chunks that overlap with any expected range
        relevant_retrieved = 0
        for chunk, score in results:
            if chunk.filepath != q["filepath"]:
                continue
            chunk_lines = set(range(chunk.start_line, chunk.end_line + 1))
            for start, end in expected_ranges:
                expected_set = set(range(start, end + 1))
                if len(chunk_lines & expected_set) > 0:
                    relevant_retrieved += 1
                    break

        precision = relevant_retrieved / top_k if top_k > 0 else 0.0

        total_precision += precision
        total_recall += recall
        details.append({
            "query": q["query"],
            "precision": precision,
            "recall": recall,
            "top_result_score": results[0][1] if results else 0.0,
        })

    n = len(queries)
    return {
        "avg_precision": total_precision / n if n else 0.0,
        "avg_recall": total_recall / n if n else 0.0,
        "num_queries": n,
        "details": details,
    }
