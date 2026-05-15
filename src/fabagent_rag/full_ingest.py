from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import sqlite3

from fabagent_rag.chunking import Chunk, ChunkConfig, batch, split_text
from fabagent_rag.config import Settings
from fabagent_rag.documents import discover_supported_documents, parse_document
from fabagent_rag.embeddings import EmbeddingModel
from fabagent_rag.keyword_store import KeywordStore
from fabagent_rag.milvus_store import MilvusStore, build_source_filter_expr
from fabagent_rag.rag_service import build_chunk_config, enrich_document_metadata, reset_indexes, summarize_error


ProgressCallback = Callable[[str, Path, int, int, str], None]


@dataclass(slots=True)
class IngestSession:
    """全量同步专用的客户端集合，只在 `ingest-all` 链路里使用。"""

    embedder: EmbeddingModel
    store: MilvusStore
    keyword_store: KeywordStore


def build_embedder(settings: Settings) -> EmbeddingModel:
    return EmbeddingModel(
        settings.embedding_model,
        settings.embedding_api_key,
        settings.embedding_base_url,
    )


def build_ingest_session(settings: Settings) -> IngestSession:
    embedder = build_embedder(settings)
    return IngestSession(
        embedder=embedder,
        store=MilvusStore(settings.milvus_host, settings.milvus_port, settings.milvus_collection, embedder.dimension),
        keyword_store=KeywordStore(settings.keyword_index_path),
    )


def count_milvus_chunks(store: MilvusStore, source: str) -> int:
    if not store.client.has_collection(store.collection_name):
        return 0

    store.validate_collection_schema()
    store.client.load_collection(store.collection_name)
    rows = store.client.query(
        collection_name=store.collection_name,
        filter=build_source_filter_expr([source]),
        output_fields=["chunk_id"],
    )
    return len(rows)


def count_keyword_chunks(keyword_store: KeywordStore, source: str) -> int:
    if not keyword_store.db_path.exists():
        return 0

    keyword_store.ensure_index()
    with keyword_store.connect() as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT COUNT(*) AS chunk_count FROM chunks_fts WHERE source = ?",
            (source,),
        ).fetchone()
    return int(row["chunk_count"] or 0) if row else 0


def delete_milvus_chunks(store: MilvusStore, source: str) -> int:
    if not store.client.has_collection(store.collection_name):
        return 0

    store.validate_collection_schema()
    store.client.load_collection(store.collection_name)
    result = store.client.delete(
        collection_name=store.collection_name,
        filter=build_source_filter_expr([source]),
    )
    store.client.flush(collection_name=store.collection_name)
    return int(result.get("delete_count") or 0)


def delete_keyword_chunks(keyword_store: KeywordStore, source: str) -> int:
    if not keyword_store.db_path.exists():
        return 0

    keyword_store.ensure_index()
    with keyword_store.connect() as connection:
        connection.execute("DELETE FROM chunks_fts WHERE source = ?", (source,))
        row = connection.execute("SELECT changes() AS deleted_count").fetchone()
    return int(row["deleted_count"] or 0) if row else 0


def build_source_action(expected_chunk_count: int, milvus_chunk_count: int, keyword_chunk_count: int) -> str:
    if expected_chunk_count <= 0:
        return "clear" if milvus_chunk_count or keyword_chunk_count else "skip_empty"
    if milvus_chunk_count == expected_chunk_count and keyword_chunk_count == expected_chunk_count:
        return "skip_complete"
    if milvus_chunk_count or keyword_chunk_count:
        return "replace"
    return "insert"


def report_progress(
    callback: ProgressCallback | None,
    stage: str,
    path: Path,
    current: int,
    total: int,
    detail: str = "",
) -> None:
    if callback:
        callback(stage, path, current, total, detail)


def sync_source(
    session: IngestSession,
    source: str,
    chunks: list[Chunk],
    batch_size: int,
    progress_callback: ProgressCallback | None = None,
    current: int = 0,
    total: int = 0,
) -> dict[str, int | str]:
    expected_chunk_count = len(chunks)
    milvus_chunk_count = count_milvus_chunks(session.store, source)
    keyword_chunk_count = count_keyword_chunks(session.keyword_store, source)
    action = build_source_action(expected_chunk_count, milvus_chunk_count, keyword_chunk_count)
    path = Path(source)

    if action == "skip_complete":
        report_progress(progress_callback, "已入库", path, current, total, f"{expected_chunk_count} 个分块")
        return {
            "documents": 1,
            "chunks": expected_chunk_count,
            "inserted": 0,
            "keyword_indexed": 0,
            "status": "skipped",
            "deleted_records": 0,
        }

    deleted_records = 0
    if action in {"replace", "clear"}:
        deleted_records += delete_milvus_chunks(session.store, source)
        deleted_records += delete_keyword_chunks(session.keyword_store, source)
        report_progress(
            progress_callback,
            "重建",
            path,
            current,
            total,
            f"已删除 {deleted_records} 条旧记录",
        )

    if action == "clear":
        report_progress(progress_callback, "跳过", path, current, total, "空内容")
        return {
            "documents": 1,
            "chunks": 0,
            "inserted": 0,
            "keyword_indexed": 0,
            "status": "cleared",
            "deleted_records": deleted_records,
        }

    if action == "skip_empty" or not chunks:
        report_progress(progress_callback, "跳过", path, current, total, "空内容")
        return {
            "documents": 1,
            "chunks": 0,
            "inserted": 0,
            "keyword_indexed": 0,
            "status": "empty",
            "deleted_records": deleted_records,
        }

    inserted = 0
    for batch_index, chunk_batch in enumerate(batch(chunks, batch_size), start=1):
        batch_total = max(1, (len(chunks) + batch_size - 1) // batch_size)
        report_progress(
            progress_callback,
            "向量化",
            path,
            current,
            total,
            f"{batch_index}/{batch_total}",
        )
        embeddings = session.embedder.encode([chunk.text for chunk in chunk_batch])
        report_progress(
            progress_callback,
            "入库",
            path,
            current,
            total,
            f"{batch_index}/{batch_total}",
        )
        inserted += session.store.insert(chunk_batch, embeddings)

    keyword_indexed = session.keyword_store.insert(chunks)
    report_progress(progress_callback, "完成", path, current, total, f"{expected_chunk_count} 个分块")
    return {
        "documents": 1,
        "chunks": expected_chunk_count,
        "inserted": inserted,
        "keyword_indexed": keyword_indexed,
        "status": "inserted" if action == "insert" else "replaced",
        "deleted_records": deleted_records,
    }


def ingest_directory(
    settings: Settings,
    directory: Path,
    batch_size: int,
    chunk_config: ChunkConfig | None = None,
    reset: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    files = discover_supported_documents(directory)
    errors: list[dict[str, str]] = []
    config = chunk_config or build_chunk_config(settings)

    if reset:
        reset_indexes(settings)

    if not files:
        return {
            "documents": 0,
            "chunks": 0,
            "inserted": 0,
            "keyword_indexed": 0,
            "scanned_files": 0,
            "parsed_files": 0,
            "failed_files": 0,
            "skipped_files": 0,
            "replaced_files": 0,
            "empty_files": 0,
            "cleared_files": 0,
            "errors": [],
        }

    session = build_ingest_session(settings)
    parsed_files = 0
    inserted = 0
    keyword_indexed = 0
    chunk_total = 0
    skipped_files = 0
    replaced_files = 0
    empty_files = 0
    cleared_files = 0

    for file_index, path in enumerate(files, start=1):
        report_progress(progress_callback, "解析", path, file_index, len(files))
        try:
            parsed_document = parse_document(path, str(path), mineru_backend=settings.mineru_backend)
        except Exception as exc:  # noqa: BLE001 - 批量入库不能因为单文件失败直接停掉
            error_message = summarize_error(exc)
            errors.append({"source": str(path), "error": error_message})
            report_progress(progress_callback, "失败", path, file_index, len(files), error_message)
            continue

        parsed_files += 1
        chunks = split_text(
            parsed_document.text,
            parsed_document.source,
            config,
            metadata=enrich_document_metadata(parsed_document),
        )
        report_progress(
            progress_callback,
            "解析完成",
            path,
            file_index,
            len(files),
            f"{len(parsed_document.text)} 字符",
        )
        report_progress(
            progress_callback,
            "切块完成",
            path,
            file_index,
            len(files),
            f"{len(chunks)} 个分块",
        )
        chunk_total += len(chunks)

        sync_result = sync_source(
            session,
            str(path),
            chunks,
            batch_size,
            progress_callback=progress_callback,
            current=file_index,
            total=len(files),
        )
        inserted += int(sync_result["inserted"])
        keyword_indexed += int(sync_result["keyword_indexed"])
        if sync_result["status"] == "skipped":
            skipped_files += 1
        elif sync_result["status"] == "replaced":
            replaced_files += 1
        elif sync_result["status"] == "empty":
            empty_files += 1
        elif sync_result["status"] == "cleared":
            cleared_files += 1

    return {
        "documents": parsed_files,
        "chunks": chunk_total,
        "inserted": inserted,
        "keyword_indexed": keyword_indexed,
        "scanned_files": len(files),
        "parsed_files": parsed_files,
        "failed_files": len(errors),
        "skipped_files": skipped_files,
        "replaced_files": replaced_files,
        "empty_files": empty_files,
        "cleared_files": cleared_files,
        "errors": errors,
    }
