"""cAST: Chunking via Abstract Syntax Trees.

Implements the recursive split-then-merge algorithm from:
  Zhang et al., "cAST: Enhancing Code Retrieval-Augmented Generation
  with Structural Chunking via Abstract Syntax Tree" (EMNLP 2025)
  https://arxiv.org/abs/2506.15655

Chunk size is measured by non-whitespace character count, following the paper.
"""

import tree_sitter
import tree_sitter_python as tspython
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    """A single chunk of code with metadata."""
    content: str
    start_line: int
    end_line: int
    filepath: str
    chunk_type: str  # "cast" or "fixed"
    nws_size: int  # non-whitespace character count

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", "\\n")
        return (
            f"Chunk({self.filepath}:{self.start_line}-{self.end_line}, "
            f"nws={self.nws_size}, '{preview}...')"
        )


def _nws_count(text: str) -> int:
    """Count non-whitespace characters in a string."""
    return sum(1 for c in text if not c.isspace())


def _node_text(node: tree_sitter.Node, source_bytes: bytes) -> str:
    """Extract the source text for a tree-sitter node."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8")


def _node_nws(node: tree_sitter.Node, source_bytes: bytes) -> int:
    """Non-whitespace char count for a node."""
    return _nws_count(_node_text(node, source_bytes))


def _chunk_nodes(
    nodes: list,
    source_bytes: bytes,
    max_chunk_size: int,
) -> List[List]:
    """Recursive split-then-merge algorithm (Algorithm 1 from the paper).

    Greedily packs AST nodes into chunks. If a node exceeds the size limit,
    recursively descend into its children. Adjacent small siblings are merged.
    """
    chunks = []
    current_chunk = []
    current_size = 0

    for node in nodes:
        node_size = _node_nws(node, source_bytes)

        if (not current_chunk and node_size > max_chunk_size) or \
           (current_size + node_size > max_chunk_size):
            # Flush current chunk if non-empty
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0

            # If this single node is too big, recurse into children
            if node_size > max_chunk_size:
                children = node.children
                if children:
                    sub_chunks = _chunk_nodes(children, source_bytes, max_chunk_size)
                    chunks.extend(sub_chunks)
                else:
                    # Leaf node that's too big — keep it as-is
                    chunks.append([node])
                continue

        # Node fits in current chunk — add it
        current_chunk.append(node)
        current_size += node_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def cast_chunk(
    code: str,
    filepath: str = "<unknown>",
    max_chunk_size: int = 2000,
) -> List[Chunk]:
    """Chunk Python source code using the cAST algorithm.

    Args:
        code: Python source code string.
        filepath: File path for metadata.
        max_chunk_size: Max non-whitespace characters per chunk.

    Returns:
        List of Chunk objects preserving syntactic boundaries.
    """
    parser = tree_sitter.Parser(tree_sitter.Language(tspython.language()))
    source_bytes = code.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    # If entire file fits in one chunk, return it directly
    if _nws_count(code) <= max_chunk_size:
        return [Chunk(
            content=code,
            start_line=1,
            end_line=code.count("\n") + 1,
            filepath=filepath,
            chunk_type="cast",
            nws_size=_nws_count(code),
        )]

    # Run recursive split-then-merge on top-level children
    node_groups = _chunk_nodes(root.children, source_bytes, max_chunk_size)

    # Convert node groups to Chunk objects
    chunks = []
    for group in node_groups:
        start_byte = group[0].start_byte
        end_byte = group[-1].end_byte
        content = source_bytes[start_byte:end_byte].decode("utf-8")
        start_line = group[0].start_point.row + 1  # 1-indexed
        end_line = group[-1].end_point.row + 1

        chunks.append(Chunk(
            content=content,
            start_line=start_line,
            end_line=end_line,
            filepath=filepath,
            chunk_type="cast",
            nws_size=_nws_count(content),
        ))

    return chunks
