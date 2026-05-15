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


PROGRESS_BAR_WIDTH = 24


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
    """把单文件处理过程渲染成会覆盖的短暂状态块。"""

    def __init__(self) -> None:
        self._rendered_lines = 0
        self._interactive = sys.stdout.isatty()

    def __call__(self, stage: str, path: Path, current: int, total: int, detail: str = "") -> None:
        lines = [format_progress_line(stage, path, current, total, detail)]
        if stage not in {"完成"}:
            lines.append(f"  文件: {path.name}")
            lines.append(f"  阶段: {stage}")
            if detail:
                lines.append(f"  细节: {detail}")
        self._render(lines)

    def _render(self, lines: list[str]) -> None:
        self._clear()
        if self._interactive:
            for line in lines:
                sys.stdout.write(f"{line}\n")
            sys.stdout.flush()
            self._rendered_lines = len(lines)
            return

        for line in lines:
            click.echo(line)
        self._rendered_lines = 0

    def _clear(self) -> None:
        if not self._interactive or self._rendered_lines <= 0:
            return

        for _ in range(self._rendered_lines):
            sys.stdout.write("\x1b[1A\x1b[2K")
        sys.stdout.flush()
        self._rendered_lines = 0


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
def ingest_all(
    directory: Path,
    batch_size: int,
    keep_old: bool,
) -> None:
    """将目录下所有支持的文档批量入库。"""

    settings = load_settings()
    renderer = IngestProgressRenderer()
    try:
        result = ingest_directory(
            settings,
            directory,
            batch_size=batch_size,
            reset=not keep_old,
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
