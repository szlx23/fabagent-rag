from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile


TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}
MINERU_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".docx", ".pptx", ".xlsx"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | MINERU_EXTENSIONS

# MinerU 会生成 Markdown、图片和中间 JSON。这里使用项目内的缓存目录，
# 并在 .gitignore 中忽略，避免把解析产物提交到仓库。
MINERU_CACHE_DIR = Path("data/mineru")


def load_documents(path: Path, pattern: str) -> list[tuple[str, str]]:
    """加载一个文件或目录，返回 `(source, text)` 列表。

    - `source` 会写入 Milvus，后续检索结果靠它告诉用户答案来自哪里。
    - `text` 是已经可分块的纯文本/Markdown；复杂文档会先交给 MinerU 解析。
    """

    if path.is_file():
        return [(str(path), load_document_text(path))]

    documents: list[tuple[str, str]] = []
    for file_path in sorted(path.glob(pattern)):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append((str(file_path), load_document_text(file_path)))
    return documents


def load_document_text(path: Path) -> str:
    """把单个文档转换为可入库文本。

    简单文本格式直接读取；PDF、图片、Office 文档统一走 MinerU，把版面解析成
    Markdown，再进入同一套 chunking/embedding 流程。
    """

    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8")
    if suffix in MINERU_EXTENSIONS:
        return parse_with_mineru(path)
    raise ValueError(f"不支持的文件类型：{path.suffix}")


def parse_with_mineru(path: Path) -> str:
    """调用 MinerU CLI，把复杂文档解析成 Markdown 文本。

    这里没有直接使用 MinerU 的内部 Python API，原因是 CLI 是它最稳定的公开
    接口之一；代价是每次解析会启动一个临时 mineru-api 服务，速度会慢一些。
    """

    mineru = find_mineru_command()
    if not mineru:
        raise RuntimeError("未找到 mineru 命令，请先在当前环境安装 MinerU。")

    MINERU_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=MINERU_CACHE_DIR) as temp_dir:
        output_dir = Path(temp_dir)
        env = os.environ.copy()
        env.setdefault("MINERU_MODEL_SOURCE", "modelscope")
        subprocess.run(
            [
                mineru,
                "-p",
                str(path),
                "-o",
                str(output_dir),
                # pipeline 后端适合 CPU-only 机器。不要使用默认 hybrid-auto-engine，
                # 默认后端更偏向高精度本地模型，可能触发更重的计算依赖。
                "-b",
                "pipeline",
                # 公式解析通常会明显增加耗时。当前 RAG 目标是术语/文档问答，
                # 先关闭公式，表格保留为 Markdown，有利于检索。
                "--formula",
                "false",
                "--table",
                "true",
            ],
            check=True,
            env=env,
        )

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
