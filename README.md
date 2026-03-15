# cAST RAG Experiment

A simple experiment demonstrating **cAST (Chunking via Abstract Syntax Trees)** for code Retrieval-Augmented Generation, based on:

> Zhang et al., "cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree" (EMNLP 2025)
> [arXiv:2506.15655](https://arxiv.org/abs/2506.15655) | [Code](https://github.com/yilinjz/astchunk)

## What This Demonstrates

The experiment compares two chunking strategies for code RAG:

1. **cAST** — Uses tree-sitter to parse code into an AST, then recursively splits large nodes and merges small siblings to produce structure-aware chunks aligned with syntactic boundaries (functions, classes, etc.)

2. **Fixed-size** — Naive line-based chunking that splits every N lines regardless of code structure.

### Key Findings (reproduced)

| Metric | cAST | Fixed | Delta |
|--------|------|-------|-------|
| Avg Recall@3 | 0.8667 | 0.8000 | **+0.0667** |
| Mid-block breaks | **0** | 20 | -20 |

- cAST produces **zero** chunks that start mid-function/mid-class
- Fixed chunking produces **20/28** chunks that break syntactic boundaries
- cAST achieves higher recall because complete functions are retrieved intact

## Project Structure

```
├── run_experiment.py          # Main experiment runner
├── src/
│   ├── cast_chunker.py        # cAST implementation (Algorithm 1 from paper)
│   ├── fixed_chunker.py       # Fixed-size baseline chunker
│   └── rag_pipeline.py        # TF-IDF retriever + evaluation
├── test_codebase/             # Synthetic Python files used as test data
│   ├── stats.py               # Statistical analysis module
│   ├── data_loader.py         # CSV/JSON data loading utilities
│   └── cache.py               # Thread-safe LRU cache
└── README.md
```

## Running

```bash
pip install tree-sitter tree-sitter-python numpy scikit-learn
python run_experiment.py
```

## How cAST Works

From the paper's Algorithm 1:

1. Parse source code into an AST using tree-sitter
2. If the entire file fits in one chunk (≤ `max_chunk_size` non-whitespace chars), return it
3. Otherwise, greedily pack top-level AST nodes into chunks
4. If a node exceeds the limit, recursively descend into its children
5. Merge adjacent small sibling nodes to maximize information density

Chunk size is measured by **non-whitespace character count** (not lines), ensuring text-dense, comparable chunks across coding styles.
