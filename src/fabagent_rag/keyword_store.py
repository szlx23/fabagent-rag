from pathlib import Path
import re
import sqlite3

from fabagent_rag.chunking import Chunk, build_chunk_id


_ASCII_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


class KeywordStore:
    """基于 SQLite FTS5 的本地 BM25 关键词索引。

    Milvus 负责向量相似度，SQLite FTS5 负责关键词和编号类精确匹配。BM25 索引
    写在本地文件中，跟随入库流程同步更新，后续混合检索可以直接融合两路结果。
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)

    def ensure_index(self) -> None:
        """创建 FTS5 表。"""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    source UNINDEXED,
                    page UNINDEXED,
                    section_title,
                    file_ext UNINDEXED,
                    content_type UNINDEXED,
                    sheet_name,
                    parser UNINDEXED,
                    ingested_at UNINDEXED,
                    text,
                    keyword_text
                )
                """
            )

    def insert(self, chunks: list[Chunk]) -> int:
        """把 chunk 写入关键词索引。"""

        if not chunks:
            return 0

        self.ensure_index()
        rows = [
            (
                chunk.chunk_id or build_chunk_id(chunk),
                chunk.source,
                chunk.page or 0,
                chunk.section_title,
                chunk.file_ext,
                chunk.content_type,
                chunk.sheet_name,
                chunk.parser,
                chunk.ingested_at,
                chunk.text,
                build_keyword_text(chunk),
            )
            for chunk in chunks
        ]
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO chunks_fts(
                    chunk_id,
                    source,
                    page,
                    section_title,
                    file_ext,
                    content_type,
                    sheet_name,
                    parser,
                    ingested_at,
                    text,
                    keyword_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def delete_source(self, source: str) -> int:
        """删除某个 source 的所有关键词索引记录。"""

        if not self.db_path.exists():
            return 0

        self.ensure_index()
        with self.connect() as connection:
            connection.execute("DELETE FROM chunks_fts WHERE source = ?", (source,))
            row = connection.execute("SELECT changes() AS deleted_count").fetchone()
        return int(row["deleted_count"] or 0) if row else 0

    def search(
        self,
        query: str,
        top_k: int,
        source_filter: list[str] | None = None,
    ) -> list[dict[str, object]]:
        """使用 FTS5 BM25 检索关键词相关 chunk。"""

        self.ensure_index()
        match_query = build_match_query(query)
        if not match_query:
            return []

        source_filter = normalize_source_filter(source_filter)
        source_clause, source_params = build_source_filter_clause(source_filter)
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"""
                SELECT
                    chunk_id,
                    source,
                    page,
                    section_title,
                    file_ext,
                    content_type,
                    sheet_name,
                    parser,
                    ingested_at,
                    text,
                    bm25(chunks_fts) AS bm25_score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                {source_clause}
                ORDER BY bm25_score
                LIMIT ?
                """,
                (match_query, *source_params, top_k),
            ).fetchall()

        matches = []
        for rank, row in enumerate(rows, start=1):
            matches.append(
                {
                    "keyword_score": 1.0 / rank,
                    "bm25_score": float(row["bm25_score"]),
                    "source": row["source"],
                    "page": normalize_keyword_page(row["page"]),
                    "section_title": row["section_title"] or "",
                    "file_ext": row["file_ext"] or "",
                    "content_type": row["content_type"] or "",
                    "sheet_name": row["sheet_name"] or "",
                    "parser": row["parser"] or "",
                    "chunk_id": row["chunk_id"] or "",
                    "ingested_at": row["ingested_at"] or "",
                    "text": row["text"],
                }
            )
        return matches

    def list_documents(self) -> list[dict[str, object]]:
        """按 source 聚合已入库文档。

        前端展示的是“知识库中可检索的资料”，所以这里从 BM25 索引聚合，
        不直接扫描 data/raw，避免把未入库或入库失败的文件展示出去。
        """

        if not self.db_path.exists():
            return []

        self.ensure_index()
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    source,
                    COUNT(*) AS chunk_count,
                    MAX(file_ext) AS file_ext,
                    MAX(parser) AS parser,
                    MAX(ingested_at) AS ingested_at
                FROM chunks_fts
                GROUP BY source
                ORDER BY source
                """
            ).fetchall()

        return [
            {
                "source": row["source"],
                "file_name": Path(row["source"]).name,
                "chunk_count": int(row["chunk_count"] or 0),
                "file_ext": row["file_ext"] or "",
                "parser": row["parser"] or "",
                "ingested_at": row["ingested_at"] or "",
            }
            for row in rows
        ]

    def drop_index(self) -> bool:
        """删除本地关键词索引文件。"""

        if self.db_path.exists():
            self.db_path.unlink()
            return True
        return False

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)


def build_keyword_text(chunk: Chunk) -> str:
    """构造适合 FTS5 的索引文本。

    SQLite 默认 tokenizer 对中文没有专门分词，所以这里额外写入中文 bigram/trigram，
    让“危险化学品”“光刻工艺”这类中文查询也能有关键词命中。
    """

    parts = [
        chunk.source,
        chunk.section_title,
        chunk.sheet_name,
        chunk.content_type,
        chunk.text,
    ]
    combined = "\n".join(part for part in parts if part)
    return f"{combined}\n{' '.join(extract_search_terms(combined))}"


def build_match_query(query: str) -> str:
    """把用户 query 转换成 FTS5 MATCH 表达式。"""

    terms = extract_search_terms(query)
    if not terms:
        return ""
    return " OR ".join(quote_fts_term(term) for term in terms[:32])


def extract_search_terms(text: str) -> list[str]:
    """抽取英文/编号 token 和中文 ngram。"""

    terms: list[str] = []
    seen = set()

    def add(term: str) -> None:
        normalized = term.strip().lower()
        if len(normalized) < 2 or normalized in seen:
            return
        terms.append(normalized)
        seen.add(normalized)

    for match in _ASCII_TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        add(token)
        compact = re.sub(r"[^A-Za-z0-9]", "", token)
        add(compact)

    for match in _CJK_PATTERN.finditer(text):
        cjk_text = match.group(0)
        for size in (2, 3):
            for index in range(0, max(0, len(cjk_text) - size + 1)):
                add(cjk_text[index : index + size])

    return terms


def quote_fts_term(term: str) -> str:
    """转义 FTS5 查询 term。"""

    return '"' + term.replace('"', '""') + '"'


def normalize_source_filter(source_filter: list[str] | None) -> list[str]:
    if not source_filter:
        return []
    return sorted({source for source in source_filter if source.strip()})


def build_source_filter_clause(source_filter: list[str]) -> tuple[str, list[str]]:
    """构造 SQLite source 过滤条件。"""

    if not source_filter:
        return "", []
    placeholders = ", ".join("?" for _ in source_filter)
    return f"AND source IN ({placeholders})", source_filter


def normalize_keyword_page(value: object) -> int | None:
    if not isinstance(value, int) or value <= 0:
        return None
    return value
