from pathlib import Path
import sys

import click

if __package__ in {None, ""}:
    # 允许 `python src/fabagent_rag/cli.py` 这种直接运行方式。
    # 正式使用仍推荐 `pip install -e .` 后执行 `rag ...`。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabagent_rag.config import load_settings
from fabagent_rag.full_ingest import ingest_directory as ingest_directory_full_sync
from fabagent_rag.evaluation import DEFAULT_EVAL_SET, DEFAULT_INTERMEDIATE_DIR, run_evaluation
from fabagent_rag.milvus_store import MilvusSchemaError
from fabagent_rag.rag_service import answer_question, ingest_path


PROGRESS_BAR_WIDTH = 24
FINAL_PROGRESS_STAGES = {"完成", "已入库", "跳过", "失败"}


def format_progress_line(stage: str, path: Path, current: int, total: int, detail: str = "") -> str:
    """把全量入库的状态压成一条可刷新的终端进度线。"""

    filled = 0 if total <= 0 else int(PROGRESS_BAR_WIDTH * current / total)
    filled = max(0, min(PROGRESS_BAR_WIDTH, filled))
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    name = path.name
    if len(name) > 34:
        name = f"{name[:15]}...{name[-14:]}"
    suffix = f" {detail}" if detail else ""
    return f"全量入库 [{bar}] {current}/{total} {stage}: {name}{suffix}"[:140].ljust(140)


class IngestProgressRenderer:
    """把单文件处理过程压成单行进度，减少全量入库时的刷屏。"""

    def __init__(self) -> None:
        self._interactive = sys.stdout.isatty()
        self._line_width = 0

    def __call__(self, stage: str, path: Path, current: int, total: int, detail: str = "") -> None:
        line = format_progress_line(stage, path, current, total, detail)
        if self._interactive:
            padded = line.ljust(max(self._line_width, len(line)))
            sys.stdout.write("\r" + padded)
            sys.stdout.flush()
            self._line_width = len(padded)
            if stage in FINAL_PROGRESS_STAGES:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._line_width = 0
            return
        click.echo(line)


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
    "--reset/--keep-old",
    default=False,
    help="默认增量同步已入库内容；需要重建 Milvus 和 BM25 索引时使用 --reset。",
)
def ingest_all(
    directory: Path,
    batch_size: int,
    reset: bool,
) -> None:
    """将目录下所有支持的文档批量同步入库。"""

    settings = load_settings()
    renderer = IngestProgressRenderer()
    try:
        result = ingest_directory_full_sync(
            settings,
            directory,
            batch_size=batch_size,
            reset=reset,
            progress_callback=renderer,
        )
    except MilvusSchemaError as exc:
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise click.ClickException(str(exc)) from exc

    sys.stdout.write("\n")
    sys.stdout.flush()
    click.echo(
        f"已扫描 {result['scanned_files']} 个文件，解析 {result['parsed_files']} 个，"
        f"完整跳过 {result.get('skipped_files', 0)} 个，重建 {result.get('replaced_files', 0)} 个，"
        f"空内容 {result.get('empty_files', 0)} 个，清理 {result.get('cleared_files', 0)} 个，"
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
@click.option(
    "--intermediate-dir",
    default=str(DEFAULT_INTERMEDIATE_DIR),
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="复用 parse/chunk 中间结果目录。",
)
def eval(
    eval_set: Path,
    stages: str,
    output_dir: Path | None,
    case_limit: int | None,
    source_limit: int | None,
    top_k_override: int | None,
    intermediate_dir: Path | None,
) -> None:
    """执行离线评测并生成报告。"""

    stage_names = tuple(stage.strip() for stage in stages.split(",") if stage.strip())
    settings = load_settings()
    try:
        run_evaluation(
            settings,
            eval_set_path=eval_set,
            stages=stage_names,
            output_dir=output_dir,
            case_limit=case_limit,
            source_limit=source_limit,
            top_k_override=top_k_override,
            intermediate_dir=intermediate_dir if intermediate_dir else None,
            progress_callback=click.echo,
        )
    except (MilvusSchemaError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
