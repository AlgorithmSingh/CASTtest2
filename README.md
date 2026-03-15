# cAST RAG Experiment

A simple experiment demonstrating **cAST (Chunking via Abstract Syntax Trees)** for code Retrieval-Augmented Generation, based on:

> Zhang et al., "cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree" (EMNLP 2025)
> [arXiv:2506.15655](https://arxiv.org/abs/2506.15655) | [Code](https://github.com/yilinjz/astchunk)

## What This Demonstrates

The experiment compares **three** chunking strategies for code RAG:

1. **cAST (ours)** — Our implementation of Algorithm 1 from the paper, using tree-sitter to parse code into an AST, then recursively splitting large nodes and merging small siblings.

2. **cAST (ref)** — The official [`astchunk`](https://github.com/yilinjz/astchunk) library by the paper authors, used as the reference implementation via `ASTChunkBuilder`.

3. **Fixed-size** — Naive line-based chunking that splits every N lines regardless of code structure (the standard RAG baseline).

### Key Findings (reproduced)

| Strategy | Precision@3 | Recall@3 | Mid-block Breaks |
|----------|------------|----------|-----------------|
| cAST (ours) | — | — | **0** |
| cAST (ref)  | — | — | **0** |
| Fixed-size   | — | — | 20 |

- Both cAST implementations produce **zero** chunks that start mid-function/mid-class
- Fixed chunking produces **20/28** chunks that break syntactic boundaries
- cAST achieves higher recall because complete functions are retrieved intact

*(Run `python run_experiment.py` to see actual numbers.)*

## Project Structure

```
├── run_experiment.py          # Main experiment runner (3-way comparison)
├── src/
│   ├── cast_chunker.py        # Our cAST impl + wrapper for official astchunk lib
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
pip install astchunk tree-sitter tree-sitter-python numpy scikit-learn
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

## References

- **Paper**: [cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree](https://arxiv.org/abs/2506.15655)
- **Official implementation**: [github.com/yilinjz/astchunk](https://github.com/yilinjz/astchunk)
- **PyPI package**: [`astchunk`](https://pypi.org/project/astchunk/)
