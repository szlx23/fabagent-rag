from pathlib import Path
import sys

import click

if __package__ in {None, ""}:
    # 允许 `python src/fabagent_rag/cli.py` 这种直接运行方式。
    # 正式使用仍推荐 `pip install -e .` 后执行 `rag ...`。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabagent_rag.config import load_settings
from fabagent_rag.rag_service import answer_question, ingest_path


@click.group()
def main() -> None:
    """基于 Milvus 的 RAG 命令行工具。"""


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--batch-size", default=10, show_default=True, help="向量化和写入的批大小。")
def ingest(path: Path, batch_size: int) -> None:
    """将单个文档写入 Milvus。"""
    settings = load_settings()
    # CLI 只负责参数解析和输出；真正业务流程在 rag_service 中，FastAPI 也复用它。
    if not path.is_file():
        raise click.ClickException("ingest 只支持单文件；批量文件请使用前端上传。")
    result = ingest_path(settings, path, batch_size)
    click.echo(f"已从 {result['documents']} 个文档写入 {result['inserted']} 个分块。")


@main.command()
@click.argument("question")
@click.option("--top-k", default=4, show_default=True, help="检索返回的分块数量。")
def ask(question: str, top_k: int) -> None:
    """基于已入库文档发起问题。"""
    settings = load_settings()
    result = answer_question(settings, question, top_k)
    click.echo(result["answer"])


if __name__ == "__main__":
    main()
