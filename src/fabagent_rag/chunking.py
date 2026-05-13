from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """写入向量库的最小文本单元。"""

    text: str
    source: str
    index: int


def split_text(text: str, source: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """把文档文本切成适合 embedding 的分块。

    RAG 检索不是按完整文档查，而是按 chunk 查。chunk 太大时语义会变稀释，
    太小时上下文又不够。`chunk_overlap` 用来让相邻片段保留一部分上下文。
    """

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
            # 尽量在段落或句子边界切分，减少把一句话从中间截断的概率。
            # 如果当前窗口里没有足够靠后的自然边界，就按固定长度切。
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
    """按批处理 chunk，避免一次性请求 embedding 接口或 Milvus 写入过多数据。"""

    for index in range(0, len(items), size):
        yield items[index : index + size]
