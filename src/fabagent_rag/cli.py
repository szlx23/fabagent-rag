from pathlib import Path
import sys

import click

if __package__ in {None, ""}:
    # 允许 `python src/fabagent_rag/cli.py` 这种直接运行方式。
    # 正式使用仍推荐 `pip install -e .` 后执行 `rag ...`。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabagent_rag.config import load_settings
from fabagent_rag.evaluation import DEFAULT_EVAL_SET, run_evaluation
from fabagent_rag.milvus_store import MilvusSchemaError
from fabagent_rag.rag_service import answer_question, ingest_directory, ingest_path


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
    try:
        result = ingest_path(settings, path, batch_size)
    except MilvusSchemaError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"已从 {result['documents']} 个文档写入 {result['inserted']} 个分块。")


@main.command(name="ingest-all")
@click.argument(
    "directory",
    required=False,
    default=Path("data/raw"),
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--batch-size", default=10, show_default=True, help="向量化和写入的批大小。")
@click.option(
    "--keep-old",
    is_flag=True,
    help="默认会先删除旧 collection 和 BM25 索引；加此参数后保留旧数据并追加写入。",
)
@click.option(
    "--exclude-prefixed",
    is_flag=True,
    help="跳过 `exclude__` / `excelude__` 前缀文件；默认会把它们一起入库。",
)
def ingest_all(
    directory: Path,
    batch_size: int,
    keep_old: bool,
    exclude_prefixed: bool,
) -> None:
    """将目录下所有支持的文档批量入库。"""

    settings = load_settings()
    try:
        result = ingest_directory(
            settings,
            directory,
            batch_size=batch_size,
            include_excluded=not exclude_prefixed,
            reset=not keep_old,
        )
    except MilvusSchemaError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"已扫描 {result['scanned_files']} 个文件，解析 {result['parsed_files']} 个，"
        f"失败 {result['failed_files']} 个，写入 {result['inserted']} 个分块。"
    )
    if result["errors"]:
        click.echo("失败文件：")
        for item in result["errors"]:
            click.echo(f"- {item['source']}: {item['error']}")


@main.command()
@click.argument("question")
@click.option("--top-k", default=4, show_default=True, help="检索返回的分块数量。")
def ask(question: str, top_k: int) -> None:
    """基于已入库文档发起问题。"""
    settings = load_settings()
    try:
        result = answer_question(settings, question, top_k)
    except MilvusSchemaError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result["answer"])


@main.command()
@click.option(
    "--eval-set",
    default=str(DEFAULT_EVAL_SET),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="评测集 JSONL 文件路径。",
)
@click.option(
    "--stages",
    default="parse,chunk,retrieval,answer",
    show_default=True,
    help="要执行的阶段，逗号分隔。",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="评测报告输出目录；默认按时间戳自动创建。",
)
@click.option("--case-limit", type=int, help="只评测前 N 条问题，便于快速 smoke test。")
@click.option("--source-limit", type=int, help="只评测前 N 个 source，便于快速 smoke test。")
@click.option("--top-k", "top_k_override", type=int, help="覆盖评测集里的 top-k 设置。")
def eval(
    eval_set: Path,
    stages: str,
    output_dir: Path | None,
    case_limit: int | None,
    source_limit: int | None,
    top_k_override: int | None,
) -> None:
    """执行离线评测并生成报告。"""

    stage_names = tuple(stage.strip() for stage in stages.split(",") if stage.strip())
    settings = load_settings()
    try:
        report_dir = run_evaluation(
            settings,
            eval_set_path=eval_set,
            stages=stage_names,
            output_dir=output_dir,
            case_limit=case_limit,
            source_limit=source_limit,
            top_k_override=top_k_override,
        )
    except MilvusSchemaError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"评测完成，报告目录：{report_dir}")


if __name__ == "__main__":
    main()
