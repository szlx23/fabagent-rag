from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha1
import re


_MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MARKDOWN_TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_MARKDOWN_LIST_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_FENCED_CODE_MARKERS = ("```", "~~~")


@dataclass(frozen=True)
class Chunk:
    """写入向量库的最小文本单元。"""

    text: str
    source: str
    index: int
    page: int | None = None
    section_title: str = ""
    file_ext: str = ""
    content_type: str = "text"
    sheet_name: str = ""
    parser: str = ""
    chunk_id: str = ""
    ingested_at: str = ""


@dataclass(frozen=True)
class ChunkConfig:
    """分块策略配置。

    `min_chunk_size` 用来定义“小 chunk”：切分后如果某个 chunk 太短，会优先尝试
    和前后 chunk 合并，只要合并后不超过 `chunk_size`。
    """

    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int


@dataclass(frozen=True)
class TextBlock:
    """从 Markdown/text 中识别出来的语义块。

    自动切块不直接按字符窗口切，而是先识别标题、段落、列表、表格、代码块。
    每个 block 携带当时的标题栈，后面生成 chunk 时可直接作为 section_title。
    """

    text: str
    section_title: str


@dataclass(frozen=True)
class ChunkDraft:
    """尚未编号入库的 chunk 草稿。"""

    text: str
    section_title: str


def split_text(
    text: str,
    source: str,
    config: ChunkConfig,
    metadata: dict[str, str] | None = None,
) -> list[Chunk]:
    """把文档文本切成适合 embedding 的分块。

    这里采用“结构优先”的切法：先按 Markdown 结构识别语义块，再把语义块打包
    成不超过 `chunk_size` 的 chunk。只有单个语义块过大时，才用字符窗口兜底切分。
    这样比纯固定窗口更稳定，表格、列表、代码块和标题上下文更不容易被截断。
    """

    clean_text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not clean_text:
        return []

    validate_chunk_config(config)

    blocks = split_markdown_blocks(clean_text)
    chunk_drafts = pack_blocks_into_chunks(blocks, config)
    merged_chunks = merge_small_chunk_drafts(chunk_drafts, config)
    base_metadata = metadata or {}
    return [
        Chunk(
            text=chunk.text,
            source=source,
            index=index,
            section_title=chunk.section_title,
            file_ext=base_metadata.get("file_ext", ""),
            content_type=detect_content_type(chunk.text),
            sheet_name=infer_sheet_name(chunk.section_title, base_metadata),
            parser=base_metadata.get("parser", ""),
            ingested_at=base_metadata.get("ingested_at", ""),
        )
        for index, chunk in enumerate(merged_chunks)
    ]


def split_markdown_blocks(text: str) -> list[TextBlock]:
    """把 Markdown/text 切成语义块，并给每个块附上标题路径。

    工业界常见做法是先尊重文档结构边界，再考虑 token/字符限制。这里不用 Markdown
    AST，是为了保持依赖简单；规则覆盖 RAG 中最关键的结构：标题、段落、表格、
    列表和 fenced code block。
    """

    lines = text.splitlines()
    blocks: list[TextBlock] = []
    heading_stack: dict[int, str] = {}
    index = 0

    def current_section_title() -> str:
        return " / ".join(heading_stack[level] for level in sorted(heading_stack))

    def append_block(block_lines: list[str], section_title: str | None = None) -> None:
        block_text = "\n".join(block_lines).strip()
        if block_text:
            blocks.append(TextBlock(block_text, section_title or current_section_title()))

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        heading = _MARKDOWN_HEADING_PATTERN.match(stripped)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            # 低层级标题进入栈时，需要弹出同级和更深层级标题。
            heading_stack = {
                heading_level: heading_title
                for heading_level, heading_title in heading_stack.items()
                if heading_level < level
            }
            heading_stack[level] = title
            append_block([line])
            index += 1
            continue

        if is_fenced_code_start(stripped):
            fence_marker = stripped[:3]
            code_lines = [line]
            index += 1
            while index < len(lines):
                code_lines.append(lines[index])
                if lines[index].strip().startswith(fence_marker):
                    index += 1
                    break
                index += 1
            append_block(code_lines)
            continue

        if is_table_start(lines, index):
            table_lines = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_lines.append(lines[index])
                index += 1
            append_block(table_lines)
            continue

        if is_list_start(stripped):
            list_lines = [line]
            index += 1
            while index < len(lines):
                next_line = lines[index]
                next_stripped = next_line.strip()
                if not next_stripped:
                    break
                if is_special_block_start(lines, index) and not is_list_start(next_stripped):
                    break
                if is_list_start(next_stripped) or next_line.startswith((" ", "\t")):
                    list_lines.append(next_line)
                    index += 1
                    continue
                break
            append_block(list_lines)
            continue

        paragraph_lines = [line]
        index += 1
        while index < len(lines):
            next_line = lines[index]
            if not next_line.strip() or is_special_block_start(lines, index):
                break
            paragraph_lines.append(next_line)
            index += 1
        append_block(paragraph_lines)

    return blocks


def pack_blocks_into_chunks(blocks: list[TextBlock], config: ChunkConfig) -> list[ChunkDraft]:
    """把语义块打包成 chunk。

    打包规则：
    1. 不跨章节合并，避免一个 chunk 的 section_title 指向不清。
    2. 同章节内尽量合并相邻语义块，提高召回上下文完整度。
    3. 单个语义块超限时才调用长度兜底切分。
    """

    drafts: list[ChunkDraft] = []
    current_text = ""
    current_section = ""

    def flush_current() -> None:
        nonlocal current_text, current_section
        if current_text.strip():
            drafts.append(ChunkDraft(current_text.strip(), current_section))
        current_text = ""
        current_section = ""

    for block in blocks:
        block_text = block.text.strip()
        if not block_text:
            continue

        if len(block_text) > config.chunk_size:
            flush_current()
            for piece in split_long_text(block_text, config):
                drafts.append(ChunkDraft(piece, block.section_title))
            continue

        if not current_text:
            current_text = block_text
            current_section = block.section_title
            continue

        next_text = join_chunks(current_text, block_text)
        if block.section_title == current_section and len(next_text) <= config.chunk_size:
            current_text = next_text
            continue

        flush_current()
        current_text = block_text
        current_section = block.section_title

    flush_current()
    return drafts


def split_long_text(text: str, config: ChunkConfig) -> list[str]:
    """兜底切分超长语义块。

    表格、代码块或长段落有可能单块超过 `chunk_size`。这时只能按长度切，但仍尽量
    在段落/句子边界断开，并使用 `chunk_overlap` 保留局部上下文。
    """

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + config.chunk_size, len(text))
        window = text[start:end]

        if end < len(text):
            paragraph_break = window.rfind("\n\n")
            sentence_break = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            break_at = (
                paragraph_break
                if paragraph_break > config.chunk_size * 0.5
                else sentence_break
            )
            if break_at > config.chunk_size * 0.5:
                end = start + break_at + 1
                window = text[start:end]

        chunks.append(window.strip())
        if end >= len(text):
            break
        start = max(0, end - config.chunk_overlap)
    return [chunk for chunk in chunks if chunk]


def merge_small_chunk_drafts(chunks: list[ChunkDraft], config: ChunkConfig) -> list[ChunkDraft]:
    """合并过短 chunk，同时保留明确的 section_title。

    和旧的纯文本合并不同，这里只合并同一章节，或一方没有章节标题的 chunk。
    这样可以减少小片段，又不会把两个不同章节混成一个来源不清的 chunk。
    """

    validate_chunk_config(config)
    normalized = [chunk for chunk in chunks if chunk.text.strip()]
    if config.min_chunk_size == 0 or len(normalized) <= 1:
        return normalized

    merged: list[ChunkDraft] = []
    index = 0
    while index < len(normalized):
        current = normalized[index]
        if len(current.text) >= config.min_chunk_size:
            merged.append(current)
            index += 1
            continue

        if merged and can_merge_chunk_drafts(merged[-1], current, config.chunk_size):
            previous = merged[-1]
            merged[-1] = ChunkDraft(
                text=join_chunks(previous.text, current.text),
                section_title=previous.section_title or current.section_title,
            )
            index += 1
            continue

        if index + 1 < len(normalized) and can_merge_chunk_drafts(
            current, normalized[index + 1], config.chunk_size
        ):
            next_chunk = normalized[index + 1]
            merged.append(
                ChunkDraft(
                    text=join_chunks(current.text, next_chunk.text),
                    section_title=current.section_title or next_chunk.section_title,
                )
            )
            index += 2
            continue

        merged.append(current)
        index += 1

    return merged


def can_merge_chunk_drafts(left: ChunkDraft, right: ChunkDraft, chunk_size: int) -> bool:
    same_section = left.section_title == right.section_title
    missing_section = not left.section_title or not right.section_title
    return (
        (same_section or missing_section)
        and len(join_chunks(left.text, right.text)) <= chunk_size
    )


def is_special_block_start(lines: list[str], index: int) -> bool:
    stripped = lines[index].strip()
    return (
        bool(_MARKDOWN_HEADING_PATTERN.match(stripped))
        or is_fenced_code_start(stripped)
        or is_table_start(lines, index)
        or is_list_start(stripped)
    )


def is_fenced_code_start(stripped_line: str) -> bool:
    return stripped_line.startswith(_FENCED_CODE_MARKERS)


def is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return "|" in lines[index] and bool(_MARKDOWN_TABLE_SEPARATOR_PATTERN.match(lines[index + 1]))


def is_list_start(stripped_line: str) -> bool:
    return bool(_MARKDOWN_LIST_PATTERN.match(stripped_line))


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

        if index + 1 < len(normalized) and can_merge(
            current, normalized[index + 1], config.chunk_size
        ):
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


def detect_content_type(text: str) -> str:
    """粗粒度判断 chunk 内容类型，供后续混合检索和 metadata 加权使用。"""

    stripped_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not stripped_lines:
        return "text"
    if any(is_table_start(stripped_lines, index) for index in range(len(stripped_lines))):
        return "table"
    if all(_MARKDOWN_HEADING_PATTERN.match(line) for line in stripped_lines):
        return "title"
    if sum(1 for line in stripped_lines if is_list_start(line)) >= max(1, len(stripped_lines) // 2):
        return "list"
    return "text"


def infer_sheet_name(section_title: str, metadata: dict[str, str]) -> str:
    """从 Excel 转出的 `Sheet: xxx` 标题里提取 sheet 名。"""

    if metadata.get("file_ext") != ".xlsx":
        return ""
    for part in reversed([item.strip() for item in section_title.split("/")]):
        if part.lower().startswith("sheet:"):
            return part.split(":", 1)[1].strip()
    return ""


def build_chunk_id(chunk: Chunk) -> str:
    """为 chunk 生成稳定 ID，用于多路召回、BM25 和向量检索结果去重。"""

    raw = f"{chunk.source}\n{chunk.index}\n{chunk.section_title}\n{chunk.text}"
    return sha1(raw.encode("utf-8")).hexdigest()


def infer_section_title(document_text: str, chunk_text: str) -> str:
    """从 Markdown 风格标题中推断 chunk 所属章节。

    Parser 输出通常是 Markdown 或 Markdown-like 文本。这里用标题栈记录当前位置：
    `# 总章` 下的 `## 小节` 会输出 `总章 / 小节`，比单个最近标题更适合员工理解来源。
    """

    chunk_start = document_text.find(chunk_text[: min(len(chunk_text), 80)])
    if chunk_start < 0:
        chunk_start = 0

    heading_stack: dict[int, str] = {}
    scanned = 0
    in_fenced_code = False
    for line in document_text.splitlines():
        stripped = line.strip()
        if scanned > chunk_start:
            break

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fenced_code = not in_fenced_code
            scanned += len(line) + 1
            continue

        if not in_fenced_code:
            match = _MARKDOWN_HEADING_PATTERN.match(stripped)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                heading_stack = {
                    heading_level: heading_title
                    for heading_level, heading_title in heading_stack.items()
                    if heading_level < level
                }
                heading_stack[level] = title
        scanned += len(line) + 1

    return " / ".join(
        heading_stack[level]
        for level in sorted(heading_stack)
    )[:512]


def batch(items: list[Chunk], size: int) -> Iterable[list[Chunk]]:
    """按批处理 chunk，避免一次性请求 embedding 接口或 Milvus 写入过多数据。"""

    for index in range(0, len(items), size):
        yield items[index : index + size]
