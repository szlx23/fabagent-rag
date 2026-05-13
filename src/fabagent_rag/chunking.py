from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    index: int


def split_text(text: str, source: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    clean_text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not clean_text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        window = clean_text[start:end]

        if end < len(clean_text):
            paragraph_break = window.rfind("\n\n")
            sentence_break = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            break_at = paragraph_break if paragraph_break > chunk_size * 0.5 else sentence_break
            if break_at > chunk_size * 0.5:
                end = start + break_at + 1
                window = clean_text[start:end]

        chunks.append(Chunk(text=window.strip(), source=source, index=index))
        index += 1

        if end >= len(clean_text):
            break
        start = max(0, end - chunk_overlap)

    return chunks


def batch(items: list[Chunk], size: int) -> Iterable[list[Chunk]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]
