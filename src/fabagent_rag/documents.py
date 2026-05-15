from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import os
import shutil
import subprocess
import sys
import tempfile


# MinerU 会生成 Markdown、图片和中间 JSON。这里使用项目内的缓存目录，
# 并在 .gitignore 中忽略，避免把解析产物提交到仓库。
MINERU_CACHE_DIR = Path("data/mineru")


@dataclass(frozen=True)
class ParsedDocument:
    """解析后的统一中间格式。

    当前 RAG 流程只需要 `source` 和 `text`，但保留 `metadata` 是为了后续支持页码、
    sheet 名、slide 编号等来源信息，而不用改 chunking/embedding 的主流程。
    """

    source: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


class DocumentParser(Protocol):
    """所有文件解析器的统一接口。"""

    def parse(self, path: Path, source: str) -> ParsedDocument:
        """把文件解析成统一的 Markdown/text 文档。"""


class NativeTextParser:
    """TXT 这类纯文本文件直接读取，不做格式转换。"""

    def parse(self, path: Path, source: str) -> ParsedDocument:
        return ParsedDocument(
            source=source,
            text=path.read_text(encoding="utf-8"),
            metadata={"parser": "native", "file_ext": path.suffix.lower()},
        )


class MarkdownParser:
    """Markdown 文件和 TXT 一样直接读取，保留标题、列表、表格等原始标记。"""

    def parse(self, path: Path, source: str) -> ParsedDocument:
        return ParsedDocument(
            source=source,
            text=path.read_text(encoding="utf-8"),
            metadata={"parser": "markdown", "file_ext": path.suffix.lower()},
        )


class MinerUParser:
    """PDF/图片走 MinerU，保留版面解析后的 Markdown。"""

    def __init__(self, backend: str) -> None:
        self.backend = backend

    def parse(self, path: Path, source: str) -> ParsedDocument:
        return ParsedDocument(
            source=source,
            text=parse_with_mineru(path, self.backend),
            metadata={"parser": "mineru", "file_ext": path.suffix.lower()},
        )


class DoclingParser:
    """DOCX/PPTX 走 Docling，输出统一 Markdown。"""

    def parse(self, path: Path, source: str) -> ParsedDocument:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise RuntimeError("解析 DOCX/PPTX 需要安装 docling。") from exc

        result = DocumentConverter().convert(path)
        return ParsedDocument(
            source=source,
            text=result.document.export_to_markdown().strip(),
            metadata={"parser": "docling", "file_ext": path.suffix.lower()},
        )


class LegacyOfficeParser:
    """DOC/PPT 先转成新 Office 格式，再复用 Docling。

    Docling 当前直接支持 docx/pptx，没有把老式二进制 doc/ppt 列为输入格式。
    这里用 LibreOffice 做一次本地转换，让用户收集到的兼容格式也能进入同一解析链路。
    """

    def parse(self, path: Path, source: str) -> ParsedDocument:
        target_extension = ".docx" if path.suffix.lower() == ".doc" else ".pptx"
        converted_path = convert_legacy_office(path, target_extension)
        try:
            parsed = DoclingParser().parse(converted_path, source)
        finally:
            converted_path.unlink(missing_ok=True)
        return ParsedDocument(
            source=parsed.source,
            text=parsed.text,
            metadata={
                **parsed.metadata,
                "file_ext": path.suffix.lower(),
                "conversion": "libreoffice",
            },
        )


class PandasExcelParser:
    """XLSX 走 pandas，每个 sheet 转成 Markdown 表格后合并。"""

    def parse(self, path: Path, source: str) -> ParsedDocument:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("解析 XLSX 需要安装 pandas 和 openpyxl。") from exc

        sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)
        sections: list[str] = []
        for sheet_name, frame in sheets.items():
            sections.append(f"## Sheet: {sheet_name}\n\n{frame.to_markdown(index=False)}")

        return ParsedDocument(
            source=source,
            text="\n\n".join(sections).strip(),
            metadata={
                "parser": "pandas",
                "file_ext": path.suffix.lower(),
                "sheets": ",".join(sheets.keys()),
            },
        )


class HtmlParser:
    """HTML 先抽取正文内容，再统一转成 Markdown-like 文本。"""

    def parse(self, path: Path, source: str) -> ParsedDocument:
        try:
            import trafilatura
        except ImportError as exc:
            raise RuntimeError("解析 HTML 需要安装 trafilatura。") from exc

        html = path.read_text(encoding="utf-8", errors="ignore")
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            output_format="markdown",
        )
        if not text:
            raise RuntimeError(f"无法从 HTML 文件解析正文：{path}")

        return ParsedDocument(
            source=source,
            text=text.strip(),
            metadata={"parser": "trafilatura", "file_ext": path.suffix.lower()},
        )


PARSERS_BY_EXTENSION: dict[str, DocumentParser] = {
    ".txt": NativeTextParser(),
    ".md": MarkdownParser(),
    ".markdown": MarkdownParser(),
    ".doc": LegacyOfficeParser(),
    ".docx": DoclingParser(),
    ".ppt": LegacyOfficeParser(),
    ".pptx": DoclingParser(),
    ".xlsx": PandasExcelParser(),
    ".html": HtmlParser(),
    ".htm": HtmlParser(),
}

MINERU_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

SUPPORTED_EXTENSIONS = set(PARSERS_BY_EXTENSION) | MINERU_EXTENSIONS


def load_document_text(path: Path, mineru_backend: str = "pipeline") -> str:
    """把单个文档转换为可入库文本。

    它先解析成 `ParsedDocument`，再取统一文本字段。
    """

    return parse_document(path, str(path), mineru_backend=mineru_backend).text


def parse_document(path: Path, source: str, mineru_backend: str = "pipeline") -> ParsedDocument:
    """根据文件扩展名选择 Parser，并输出统一中间格式。"""

    suffix = path.suffix.lower()
    if suffix in MINERU_EXTENSIONS:
        return MinerUParser(mineru_backend).parse(path, source)

    parser = PARSERS_BY_EXTENSION.get(suffix)
    if not parser:
        raise ValueError(f"不支持的文件类型：{path.suffix}")
    return parser.parse(path, source)


def discover_supported_documents(
    directory: Path,
    include_excluded: bool = True,
) -> list[Path]:
    """扫描目录中可解析的文件。

    这里默认把 `data/raw` 下所有支持的文件都纳入，包含 `exclude__` / `excelude__`
    前缀文件；如果后续只想做日常回归，可以通过 `include_excluded=False` 收缩范围。
    """

    files = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if not include_excluded and path.name.startswith(("exclude__", "excelude__")):
            continue
        files.append(path)
    return files


def convert_legacy_office(path: Path, target_extension: str) -> Path:
    """用 LibreOffice 把 doc/ppt 转为 docx/pptx 临时文件。"""

    office = find_office_converter()
    if not office:
        raise RuntimeError("解析 DOC/PPT 需要安装 LibreOffice 或 soffice。")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        try:
            subprocess.run(
                [
                    office,
                    "--headless",
                    "--convert-to",
                    target_extension.lstrip("."),
                    "--outdir",
                    str(output_dir),
                    str(path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="ignore").strip()
            message = f"LibreOffice 转换 {path} 失败。"
            if stderr:
                message = f"{message}\n{stderr[-800:]}"
            raise RuntimeError(message) from exc

        converted_files = sorted(output_dir.glob(f"*{target_extension}"))
        if not converted_files:
            raise RuntimeError(f"LibreOffice 未能把 {path} 转换为 {target_extension}。")

        # TemporaryDirectory 会清理目录，所以复制到另一个临时文件供 Docling 读取。
        stable_file = tempfile.NamedTemporaryFile(suffix=target_extension, delete=False)
        stable_file.close()
        shutil.copyfile(converted_files[0], stable_file.name)
        return Path(stable_file.name)


def find_office_converter() -> str | None:
    """查找 LibreOffice/soffice 命令。"""

    return shutil.which("libreoffice") or shutil.which("soffice")


def parse_with_mineru(path: Path, backend: str) -> str:
    """调用 MinerU CLI，把 PDF 解析成 Markdown 文本。

    这里没有直接使用 MinerU 的内部 Python API，原因是 CLI 是它最稳定的公开接口之一；
    代价是每次解析会启动一个临时 mineru-api 服务，速度会慢一些。
    """

    mineru = find_mineru_command()
    if not mineru:
        raise RuntimeError("未找到 mineru 命令，请先在当前环境安装 MinerU。")

    MINERU_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=MINERU_CACHE_DIR) as temp_dir:
        output_dir = Path(temp_dir)
        env = os.environ.copy()
        env.setdefault("MINERU_MODEL_SOURCE", "modelscope")
        try:
            subprocess.run(
                [
                    mineru,
                    "-p",
                    str(path),
                    "-o",
                    str(output_dir),
                    "-b",
                    backend,
                    # 公式解析通常会明显增加耗时。当前 RAG 目标是术语/文档问答，
                    # 先关闭公式，表格保留为 Markdown，有利于检索。
                    "--formula",
                    "false",
                    "--table",
                    "true",
                ],
                check=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="ignore").strip()
            message = f"MinerU 解析 {path} 失败。"
            if stderr:
                message = f"{message}\n{stderr[-800:]}"
            raise RuntimeError(message) from exc

        markdown_files = sorted(output_dir.rglob("*.md"))
        if not markdown_files:
            raise RuntimeError(f"MinerU 未从 {path} 解析出 Markdown 文件。")

        return "\n\n".join(
            markdown_file.read_text(encoding="utf-8")
            for markdown_file in markdown_files
        ).strip()


def find_mineru_command() -> str | None:
    """查找当前 Python 环境中的 `mineru` 可执行文件。

    API 服务不一定继承用户 shell 的 PATH，所以先从 `sys.executable` 同目录查找。
    这能保证 `conda activate rag` 之外的启动方式也尽量找到正确环境的 MinerU。
    """

    python_bin_mineru = Path(sys.executable).with_name("mineru")
    if python_bin_mineru.exists():
        return str(python_bin_mineru)
    return shutil.which("mineru")
