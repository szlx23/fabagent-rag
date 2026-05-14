from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """写入向量库的最小文本单元。"""

    text: str
    source: str
    index: int
    page: int | None = None
    section_title: str = ""


@dataclass(frozen=True)
class ChunkConfig:
    """分块策略配置。

    `min_chunk_size` 用来定义“小 chunk”：切分后如果某个 chunk 太短，会优先尝试
    和前后 chunk 合并，只要合并后不超过 `chunk_size`。
    """

    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int


def split_text(text: str, source: str, config: ChunkConfig) -> list[Chunk]:
    """把文档文本切成适合 embedding 的分块。

    RAG 检索不是按完整文档查，而是按 chunk 查。chunk 太大时语义会变稀释，
    太小时上下文又不够。`chunk_overlap` 用来让相邻片段保留一部分上下文。
    """

    clean_text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not clean_text:
        return []

    validate_chunk_config(config)

    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap

    raw_chunks: list[str] = []
    start = 0

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

        raw_chunks.append(window.strip())

        if end >= len(clean_text):
            break
        start = max(0, end - chunk_overlap)

    merged_chunks = merge_small_text_chunks(raw_chunks, config)
    return [
        Chunk(
            text=chunk_text,
            source=source,
            index=index,
            section_title=infer_section_title(clean_text, chunk_text),
        )
        for index, chunk_text in enumerate(merged_chunks)
    ]


def validate_chunk_config(config: ChunkConfig) -> None:
    """校验分块参数，避免运行时出现无限循环或不可达的合并策略。"""

    if config.chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if config.chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0")
    if config.min_chunk_size < 0:
        raise ValueError("min_chunk_size 不能小于 0")
    if config.chunk_overlap >= config.chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    if config.min_chunk_size > config.chunk_size:
        raise ValueError("min_chunk_size 不能大于 chunk_size")


def merge_small_text_chunks(chunks: list[str], config: ChunkConfig) -> list[str]:
    """合并过短 chunk，尽量减少没有独立语义的小片段。

    合并方向优先向前，因为前一个 chunk 通常是当前短片段的上文；如果向前会超长，
    再尝试和后一个 chunk 合并。
    """

    validate_chunk_config(config)
    normalized = [chunk.strip() for chunk in chunks if chunk.strip()]
    if config.min_chunk_size == 0 or len(normalized) <= 1:
        return normalized

    merged: list[str] = []
    index = 0
    while index < len(normalized):
        current = normalized[index]

        if len(current) >= config.min_chunk_size:
            merged.append(current)
            index += 1
            continue

        if merged and can_merge(merged[-1], current, config.chunk_size):
            merged[-1] = join_chunks(merged[-1], current)
            index += 1
            continue

        if index + 1 < len(normalized) and can_merge(current, normalized[index + 1], config.chunk_size):
            merged.append(join_chunks(current, normalized[index + 1]))
            index += 2
            continue

        merged.append(current)
        index += 1

    return merged


def can_merge(left: str, right: str, chunk_size: int) -> bool:
    return len(join_chunks(left, right)) <= chunk_size


def join_chunks(left: str, right: str) -> str:
    return f"{left.rstrip()}\n\n{right.lstrip()}".strip()


def infer_section_title(document_text: str, chunk_text: str) -> str:
    """从 Markdown 风格标题中推断 chunk 所属章节。

    Parser 输出通常是 Markdown 或 Markdown-like 文本。这里选择离 chunk 最近的上方
    标题作为员工可读的来源位置；找不到标题时返回空字符串。
    """

    chunk_start = document_text.find(chunk_text[: min(len(chunk_text), 80)])
    if chunk_start < 0:
        chunk_start = 0

    section_title = ""
    scanned = 0
    for line in document_text.splitlines():
        stripped = line.strip()
        if scanned > chunk_start:
            break
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                section_title = title
        scanned += len(line) + 1
    return section_title[:512]


def batch(items: list[Chunk], size: int) -> Iterable[list[Chunk]]:
    """按批处理 chunk，避免一次性请求 embedding 接口或 Milvus 写入过多数据。"""

    for index in range(0, len(items), size):
        yield items[index : index + size]
