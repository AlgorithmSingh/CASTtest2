"""Fixed-size line-based chunking baseline.

This is the standard naive approach that most RAG systems use:
split code into fixed-size chunks by line count, with no awareness
of syntactic structure.
"""

from typing import List
from src.cast_chunker import Chunk, _nws_count


def fixed_chunk(
    code: str,
    filepath: str = "<unknown>",
    max_lines: int = 20,
) -> List[Chunk]:
    """Chunk code into fixed-size segments by line count.

    Args:
        code: Source code string.
        filepath: File path for metadata.
        max_lines: Maximum number of lines per chunk.

    Returns:
        List of Chunk objects with fixed-size splits.
    """
    lines = code.split("\n")
    chunks = []

    for i in range(0, len(lines), max_lines):
        chunk_lines = lines[i : i + max_lines]
        content = "\n".join(chunk_lines)
        if not content.strip():
            continue
        chunks.append(Chunk(
            content=content,
            start_line=i + 1,
            end_line=min(i + max_lines, len(lines)),
            filepath=filepath,
            chunk_type="fixed",
            nws_size=_nws_count(content),
        ))

    return chunks
