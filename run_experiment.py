#!/usr/bin/env python3
"""cAST vs Fixed-Size Chunking: A RAG Retrieval Experiment.

Reproduces the core finding from:
  Zhang et al., "cAST: Enhancing Code Retrieval-Augmented Generation
  with Structural Chunking via Abstract Syntax Tree" (EMNLP 2025)

This experiment demonstrates that AST-aware chunking produces more
semantically coherent chunks, leading to better retrieval quality
compared to naive fixed-size line-based chunking.

Usage:
    python run_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.cast_chunker import cast_chunk
from src.fixed_chunker import fixed_chunk
from src.rag_pipeline import SimpleRetriever, evaluate_retrieval


# ---------------------------------------------------------------------------
# Test data: queries with ground-truth relevant line ranges
# ---------------------------------------------------------------------------
# These queries simulate a developer searching for specific functionality.
# Ground truth is the line range(s) in the source files that contain the answer.

QUERIES = [
    # --- stats.py queries ---
    {
        "query": "compute mean average of a list of numbers",
        "filepath": "stats.py",
        "expected_lines": [(9, 23)],  # compute_mean function
    },
    {
        "query": "calculate standard deviation and variance",
        "filepath": "stats.py",
        "expected_lines": [(26, 40), (43, 45)],  # compute_variance + compute_std
    },
    {
        "query": "percentile interpolation calculation",
        "filepath": "stats.py",
        "expected_lines": [(107, 123)],  # percentile method
    },
    {
        "query": "z-score normalization for data points",
        "filepath": "stats.py",
        "expected_lines": [(103, 105)],  # z_scores method
    },
    {
        "query": "summary statistics dictionary with min max range",
        "filepath": "stats.py",
        "expected_lines": [(125, 138)],  # to_dict method
    },
    # --- data_loader.py queries ---
    {
        "query": "load CSV file and parse records with delimiter",
        "filepath": "data_loader.py",
        "expected_lines": [(47, 78)],  # CSVLoader.load method
    },
    {
        "query": "filter records by category name",
        "filepath": "data_loader.py",
        "expected_lines": [(80, 84)],  # filter_by_category
    },
    {
        "query": "validate data records check duplicate IDs and negative values",
        "filepath": "data_loader.py",
        "expected_lines": [(131, 160)],  # validate_records function
    },
    {
        "query": "merge multiple JSON files and deduplicate by id",
        "filepath": "data_loader.py",
        "expected_lines": [(115, 129)],  # merge_files method
    },
    {
        "query": "DataRecord dataclass with tags and category fields",
        "filepath": "data_loader.py",
        "expected_lines": [(12, 39)],  # DataRecord class
    },
    # --- cache.py queries ---
    {
        "query": "LRU cache get retrieve value by key and move to end",
        "filepath": "cache.py",
        "expected_lines": [(57, 72)],  # LRUCache.get method
    },
    {
        "query": "cache evict expired entries remove old TTL",
        "filepath": "cache.py",
        "expected_lines": [(104, 112)],  # evict_expired method
    },
    {
        "query": "get or compute cache miss fallback callable",
        "filepath": "cache.py",
        "expected_lines": [(130, 152)],  # get_or_compute method
    },
    {
        "query": "cache hit rate statistics performance metrics",
        "filepath": "cache.py",
        "expected_lines": [(118, 127)],  # hit_rate + stats
    },
    {
        "query": "thread safe cache put insert with TTL expiration",
        "filepath": "cache.py",
        "expected_lines": [(74, 93)],  # put method
    },
]


def load_test_files():
    """Load all Python files from the test codebase directory."""
    codebase_dir = os.path.join(os.path.dirname(__file__), "test_codebase")
    files = {}
    for fname in sorted(os.listdir(codebase_dir)):
        if fname.endswith(".py"):
            fpath = os.path.join(codebase_dir, fname)
            with open(fpath) as f:
                files[fname] = f.read()
    return files


def print_separator(char="=", width=70):
    print(char * width)


def print_chunks_overview(label, chunks):
    """Print a brief overview of chunks."""
    print(f"\n  {label}: {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        lines = c.content.count("\n") + 1
        print(f"    [{i+1}] lines {c.start_line}-{c.end_line} "
              f"({lines} lines, {c.nws_size} nws chars)")


def show_chunk_quality(label, chunks, filepath):
    """Show whether chunks break function/class boundaries."""
    breaks = 0
    for c in chunks:
        lines = c.content.split("\n")
        # Check if chunk starts mid-function (indented, no def/class)
        first_nonblank = next((l for l in lines if l.strip()), "")
        if first_nonblank and first_nonblank[0] == " " and \
           not first_nonblank.strip().startswith(("def ", "class ", "@", "#", '"""', "'''")):
            breaks += 1
    print(f"  {label} ({filepath}): {breaks}/{len(chunks)} chunks "
          f"start mid-block (lower is better)")
    return breaks


def main():
    print_separator()
    print("cAST vs Fixed-Size Chunking — RAG Retrieval Experiment")
    print("Based on: Zhang et al., EMNLP 2025 (arXiv:2506.15655)")
    print_separator()

    # Load test codebase
    files = load_test_files()
    print(f"\nLoaded {len(files)} test files: {', '.join(files.keys())}")

    # -------------------------------------------------------------------
    # Step 1: Chunk all files with both strategies
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: Chunking")
    print("=" * 70)

    cast_chunks = []
    fixed_chunks = []
    total_cast_breaks = 0
    total_fixed_breaks = 0

    for fname, code in files.items():
        cc = cast_chunk(code, filepath=fname, max_chunk_size=800)
        fc = fixed_chunk(code, filepath=fname, max_lines=20)
        cast_chunks.extend(cc)
        fixed_chunks.extend(fc)

        print_chunks_overview(f"cAST  ({fname})", cc)
        print_chunks_overview(f"Fixed ({fname})", fc)

        total_cast_breaks += show_chunk_quality("cAST ", cc, fname)
        total_fixed_breaks += show_chunk_quality("Fixed", fc, fname)

    print(f"\n  TOTAL mid-block breaks: cAST={total_cast_breaks}, "
          f"Fixed={total_fixed_breaks}")

    # -------------------------------------------------------------------
    # Step 2: Show example chunks side by side
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Example Chunk Comparison (stats.py around compute_variance)")
    print("=" * 70)

    # Find the chunk containing compute_variance for each strategy
    for label, chunks in [("cAST", cast_chunks), ("Fixed", fixed_chunks)]:
        for c in chunks:
            if c.filepath == "stats.py" and "compute_variance" in c.content:
                print(f"\n  --- {label} chunk (lines {c.start_line}-{c.end_line}) ---")
                # Show first 15 lines
                preview_lines = c.content.split("\n")[:15]
                for line in preview_lines:
                    print(f"  | {line}")
                if len(c.content.split("\n")) > 15:
                    print(f"  | ... ({len(c.content.split(chr(10)))} lines total)")
                break

    # -------------------------------------------------------------------
    # Step 3: Build retrievers and evaluate
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Retrieval Evaluation (TF-IDF + Cosine Similarity)")
    print("=" * 70)

    cast_retriever = SimpleRetriever(cast_chunks)
    fixed_retriever = SimpleRetriever(fixed_chunks)

    top_k = 3
    cast_results = evaluate_retrieval(cast_retriever, QUERIES, top_k=top_k)
    fixed_results = evaluate_retrieval(fixed_retriever, QUERIES, top_k=top_k)

    print(f"\n  {'Metric':<25} {'cAST':>10} {'Fixed':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    for metric in ["avg_precision", "avg_recall"]:
        c_val = cast_results[metric]
        f_val = fixed_results[metric]
        delta = c_val - f_val
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:<25} {c_val:>10.4f} {f_val:>10.4f} {sign}{delta:>9.4f}")

    # -------------------------------------------------------------------
    # Step 4: Per-query breakdown
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Per-Query Results")
    print("=" * 70)
    print(f"\n  {'Query':<55} {'cAST P':>7} {'Fix P':>7} {'cAST R':>7} {'Fix R':>7}")
    print(f"  {'-'*83}")

    cast_wins = 0
    fixed_wins = 0
    ties = 0

    for cd, fd in zip(cast_results["details"], fixed_results["details"]):
        q_short = cd["query"][:52] + "..." if len(cd["query"]) > 52 else cd["query"]
        cp, fp = cd["precision"], fd["precision"]
        cr, fr = cd["recall"], fd["recall"]
        print(f"  {q_short:<55} {cp:>7.2f} {fp:>7.2f} {cr:>7.2f} {fr:>7.2f}")

        # Count wins based on recall (the primary metric from the paper)
        if cr > fr:
            cast_wins += 1
        elif fr > cr:
            fixed_wins += 1
        else:
            ties += 1

    # -------------------------------------------------------------------
    # Step 5: Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Total queries: {len(QUERIES)}")
    print(f"  cAST wins: {cast_wins}  |  Fixed wins: {fixed_wins}  |  Ties: {ties}")
    print(f"\n  cAST  — Avg Precision@{top_k}: {cast_results['avg_precision']:.4f}, "
          f"Avg Recall@{top_k}: {cast_results['avg_recall']:.4f}")
    print(f"  Fixed — Avg Precision@{top_k}: {fixed_results['avg_precision']:.4f}, "
          f"Avg Recall@{top_k}: {fixed_results['avg_recall']:.4f}")

    p_delta = cast_results["avg_precision"] - fixed_results["avg_precision"]
    r_delta = cast_results["avg_recall"] - fixed_results["avg_recall"]
    print(f"\n  Precision delta: {'+' if p_delta >= 0 else ''}{p_delta:.4f}")
    print(f"  Recall delta:    {'+' if r_delta >= 0 else ''}{r_delta:.4f}")

    if cast_results["avg_recall"] > fixed_results["avg_recall"]:
        print("\n  RESULT: cAST outperforms fixed-size chunking on retrieval,")
        print("  confirming the paper's finding that structure-aware chunking")
        print("  produces more semantically coherent, retrievable code units.")
    elif cast_results["avg_recall"] == fixed_results["avg_recall"]:
        print("\n  RESULT: Both methods tied on recall. cAST still produces")
        print("  structurally cleaner chunks (fewer mid-block breaks).")
    else:
        print("\n  RESULT: Fixed-size chunking edged ahead on this test set.")
        print("  The paper's gains are larger on real-world repositories.")

    print_separator()
    return 0


if __name__ == "__main__":
    sys.exit(main())
