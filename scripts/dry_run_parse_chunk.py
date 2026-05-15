from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
import argparse
import csv
import json
import math
import re
import traceback

from fabagent_rag.config import load_settings
from fabagent_rag.documents import SUPPORTED_EXTENSIONS, parse_document
from fabagent_rag.rag_service import build_chunk_config, enrich_document_metadata
from fabagent_rag.chunking import split_text


DEFAULT_OUTPUT_DIR = Path("data/eval/parse_chunk_dry_run")
DEFAULT_EMBEDDING_BATCH_SIZE = 10
CHARS_PER_TOKEN_ESTIMATE = 4
MONTHLY_CALL_QUOTA = 18_000
FIVE_HOUR_CALL_QUOTA = 1_200
WEEKLY_CALL_QUOTA = 90_000


@dataclass
class FileEstimate:
    source: str
    file_name: str
    file_ext: str
    file_size_bytes: int
    status: str
    parser: str = ""
    parsed_chars: int = 0
    estimated_tokens: int = 0
    chunk_count: int = 0
    avg_chunk_chars: int = 0
    max_chunk_chars: int = 0
    embedding_batch_calls: int = 0
    error: str = ""


def main() -> None:
    args = parse_args()
    settings = load_settings()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    parsed_dir = output_dir / "parsed"
    chunks_dir = output_dir / "chunks"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(input_dir)
    estimates = []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path}", flush=True)
        if args.resume_existing:
            existing = load_existing_estimate(
                path, parsed_dir, chunks_dir, args.embedding_batch_size
            )
            if existing is not None:
                estimates.append(existing)
                print(
                    f"  skip: chunks={existing.chunk_count}, "
                    f"chars={existing.parsed_chars}, calls={existing.embedding_batch_calls}",
                    flush=True,
                )
                continue
        estimate = process_file(
            path=path,
            parsed_dir=parsed_dir,
            chunks_dir=chunks_dir,
            embedding_batch_size=args.embedding_batch_size,
            mineru_backend=settings.mineru_backend,
            chunk_config=build_chunk_config(settings),
        )
        estimates.append(estimate)
        print(
            f"  {estimate.status}: chunks={estimate.chunk_count}, "
            f"chars={estimate.parsed_chars}, calls={estimate.embedding_batch_calls}",
            flush=True,
        )

    write_outputs(estimates, output_dir, args.embedding_batch_size)
    print(f"\n已写入 dry-run 结果：{output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="解析 data/raw 文档并 dry-run chunk，用于估算 embedding 成本。"
    )
    parser.add_argument("--input-dir", default="data/raw", help="待扫描文档目录。")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="结果输出目录。")
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="估算 embedding 入库调用次数时使用的批大小。",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="如果输出目录里已经有对应的 parsed/chunks 结果，则直接读取并跳过重跑。",
    )
    return parser.parse_args()


def discover_files(input_dir: Path) -> list[Path]:
    files = []
    for path in sorted(input_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        files.append(path)
    return files


def process_file(
    path: Path,
    parsed_dir: Path,
    chunks_dir: Path,
    embedding_batch_size: int,
    mineru_backend: str,
    chunk_config,
) -> FileEstimate:
    file_estimate = FileEstimate(
        source=str(path),
        file_name=path.name,
        file_ext=path.suffix.lower(),
        file_size_bytes=path.stat().st_size,
        excluded_by_prefix=is_excluded(path),
        status="ok",
    )

    try:
        document = parse_document(path, str(path), mineru_backend=mineru_backend)
        metadata = enrich_document_metadata(document)
        chunks = split_text(document.text, document.source, chunk_config, metadata=metadata)
        chunk_lengths = [len(chunk.text) for chunk in chunks]

        file_estimate.parser = document.metadata.get("parser", "")
        file_estimate.parsed_chars = len(document.text)
        file_estimate.estimated_tokens = estimate_tokens(document.text)
        file_estimate.chunk_count = len(chunks)
        file_estimate.avg_chunk_chars = int(sum(chunk_lengths) / len(chunk_lengths)) if chunks else 0
        file_estimate.max_chunk_chars = max(chunk_lengths) if chunks else 0
        file_estimate.embedding_batch_calls = math.ceil(len(chunks) / embedding_batch_size)

        safe_name = safe_file_stem(path)
        (parsed_dir / f"{safe_name}.md").write_text(document.text, encoding="utf-8")
        with (chunks_dir / f"{safe_name}.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 - dry-run should record failures and continue.
        file_estimate.status = "error"
        file_estimate.error = f"{type(exc).__name__}: {exc}"
        error_path = chunks_dir / f"{safe_file_stem(path)}.error.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")

    return file_estimate


def write_outputs(
    estimates: list[FileEstimate],
    output_dir: Path,
    embedding_batch_size: int,
) -> None:
    rows = [asdict(estimate) for estimate in estimates]
    summary = build_summary(estimates, embedding_batch_size)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "files": rows}, handle, ensure_ascii=False, indent=2)

    with (output_dir / "files.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0]) if rows else list(FileEstimate.__dataclass_fields__)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "README.md").write_text(build_report(summary, estimates), encoding="utf-8")


def load_existing_estimate(
    path: Path,
    parsed_dir: Path,
    chunks_dir: Path,
    embedding_batch_size: int,
) -> FileEstimate | None:
    safe_name = safe_file_stem(path)
    chunk_path = chunks_dir / f"{safe_name}.jsonl"
    error_path = chunks_dir / f"{safe_name}.error.txt"
    if error_path.exists():
        error_text = error_path.read_text(encoding="utf-8")
        return FileEstimate(
            source=str(path),
            file_name=path.name,
            file_ext=path.suffix.lower(),
            file_size_bytes=path.stat().st_size,
            excluded_by_prefix=is_excluded(path),
            status="error",
            error=error_text.splitlines()[-1] if error_text else "previous error",
        )
    if not chunk_path.exists():
        return None

    chunk_lengths = []
    parser = ""
    with chunk_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not parser:
                parser = row.get("parser", "")
            chunk_lengths.append(len(row.get("text", "")))

    parsed_path = parsed_dir / f"{safe_name}.md"
    parsed_text = parsed_path.read_text(encoding="utf-8") if parsed_path.exists() else ""
    parsed_chars = len(parsed_text) if parsed_text else sum(chunk_lengths)
    return FileEstimate(
        source=str(path),
        file_name=path.name,
        file_ext=path.suffix.lower(),
        file_size_bytes=path.stat().st_size,
        excluded_by_prefix=is_excluded(path),
        status="ok",
        parser=parser,
        parsed_chars=parsed_chars,
        estimated_tokens=estimate_tokens(parsed_text) if parsed_text else estimate_tokens(" " * parsed_chars),
        chunk_count=len(chunk_lengths),
        avg_chunk_chars=int(sum(chunk_lengths) / len(chunk_lengths)) if chunk_lengths else 0,
        max_chunk_chars=max(chunk_lengths) if chunk_lengths else 0,
        embedding_batch_calls=math.ceil(len(chunk_lengths) / embedding_batch_size),
    )


def build_summary(estimates: list[FileEstimate], embedding_batch_size: int) -> dict[str, int]:
    ok_files = [estimate for estimate in estimates if estimate.status == "ok"]
    total_chunks = sum(estimate.chunk_count for estimate in ok_files)
    total_chars = sum(estimate.parsed_chars for estimate in ok_files)
    total_tokens = sum(estimate.estimated_tokens for estimate in ok_files)
    per_file_probe_calls = len(ok_files)
    batch_upload_probe_calls = 1 if ok_files else 0
    embedding_batch_calls = sum(estimate.embedding_batch_calls for estimate in ok_files)
    rag_question_calls = 4
    batch_ingest_calls = embedding_batch_calls + batch_upload_probe_calls
    per_file_ingest_calls = embedding_batch_calls + per_file_probe_calls
    return {
        "total_files": len(estimates),
        "ok_files": len(ok_files),
        "error_files": len(estimates) - len(ok_files),
        "total_chunks": total_chunks,
        "total_parsed_chars": total_chars,
        "estimated_embedding_tokens": total_tokens,
        "embedding_batch_size": embedding_batch_size,
        "embedding_batch_calls": embedding_batch_calls,
        "estimated_embedding_calls_batch_ingest": batch_ingest_calls,
        "estimated_embedding_calls_per_file_cli": per_file_ingest_calls,
        "estimated_llm_calls_per_rag_question": 3,
        "estimated_embedding_calls_per_rag_question": 1,
        "estimated_total_calls_per_rag_question": rag_question_calls,
        "monthly_call_quota": MONTHLY_CALL_QUOTA,
        "five_hour_call_quota": FIVE_HOUR_CALL_QUOTA,
        "weekly_call_quota": WEEKLY_CALL_QUOTA,
        "remaining_monthly_calls_after_batch_ingest": MONTHLY_CALL_QUOTA - batch_ingest_calls,
        "remaining_five_hour_calls_after_batch_ingest": FIVE_HOUR_CALL_QUOTA - batch_ingest_calls,
        "rag_questions_after_batch_ingest_monthly": max(0, (MONTHLY_CALL_QUOTA - batch_ingest_calls) // rag_question_calls),
        "rag_questions_after_batch_ingest_five_hour": max(0, (FIVE_HOUR_CALL_QUOTA - batch_ingest_calls) // rag_question_calls),
    }


def build_report(summary: dict[str, int], estimates: list[FileEstimate]) -> str:
    largest = sorted(
        [estimate for estimate in estimates if estimate.status == "ok"],
        key=lambda item: item.estimated_tokens,
        reverse=True,
    )[:20]
    failures = [estimate for estimate in estimates if estimate.status != "ok"]

    lines = [
        "# Parse/Chunk Dry Run",
        "",
        "该目录由 `scripts/dry_run_parse_chunk.py` 生成，只做解析和 chunk，不调用 embedding，"
        "用于估算入库成本和筛选测试文件。",
        "",
        "## 汇总",
        "",
        f"- 文件数：{summary['total_files']}",
        f"- 成功解析：{summary['ok_files']}",
        f"- 解析失败：{summary['error_files']}",
        f"- chunk 总数：{summary['total_chunks']}",
        f"- 解析后字符数：{summary['total_parsed_chars']}",
        f"- 粗略 embedding token：{summary['estimated_embedding_tokens']}",
        f"- embedding 批大小：{summary['embedding_batch_size']}",
        f"- 批量入库预计 embedding 调用：{summary['estimated_embedding_calls_batch_ingest']}",
        f"- 逐文件 CLI 入库预计 embedding 调用：{summary['estimated_embedding_calls_per_file_cli']}",
        "",
        "## 火山调用额度估算",
        "",
        f"- 月额度：{summary['monthly_call_quota']} 次；批量入库后约剩 "
        f"{summary['remaining_monthly_calls_after_batch_ingest']} 次",
        f"- 每 5 小时额度：{summary['five_hour_call_quota']} 次；批量入库后约剩 "
        f"{summary['remaining_five_hour_calls_after_batch_ingest']} 次",
        f"- 周额度：{summary['weekly_call_quota']} 次",
        f"- 批量入库占月额度：{format_pct(summary['estimated_embedding_calls_batch_ingest'], summary['monthly_call_quota'])}",
        f"- 批量入库占 5 小时额度：{format_pct(summary['estimated_embedding_calls_batch_ingest'], summary['five_hour_call_quota'])}",
        f"- 批量入库后，月额度约还能支撑 "
        f"{summary['rag_questions_after_batch_ingest_monthly']} 个 RAG 问题",
        f"- 批量入库后，每 5 小时额度约还能支撑 "
        f"{summary['rag_questions_after_batch_ingest_five_hour']} 个 RAG 问题",
        "",
        "## 每个 RAG 问题的调用估算",
        "",
        "- 对话模型：约 3 次，分别是 intent、query plan、answer",
        "- embedding：约 1 次，用于 query plan 中多个 query 的批量 embedding",
        "",
        "## token 最高的文件",
        "",
        "| 文件 | chunks | 估算 tokens | parser |",
        "| --- | ---: | ---: | --- |",
    ]
    for estimate in largest:
        lines.append(
            f"| `{estimate.file_name}` | {estimate.chunk_count} | "
            f"{estimate.estimated_tokens} | {estimate.parser} |"
        )

    if failures:
        lines.extend(["", "## 解析失败", "", "| 文件 | 错误 |", "| --- | --- |"])
        for estimate in failures:
            lines.append(f"| `{estimate.file_name}` | {estimate.error} |")

    lines.append("")
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE)


def safe_file_stem(path: Path) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._-") or "document"
    digest = sha1(path.name.encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{digest}"


def format_pct(part: int, whole: int) -> str:
    if whole <= 0:
        return "0.00%"
    return f"{part / whole:.2%}"


if __name__ == "__main__":
    main()
