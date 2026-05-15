from pathlib import Path
from typing import Annotated
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from fabagent_rag.config import Settings, load_settings
from fabagent_rag.documents import ParsedDocument, parse_document
from fabagent_rag.rag_service import (
    answer_question,
    build_chunk_config,
    ingest_documents,
    ingest_manual_chunks,
    ingest_path,
)

app = FastAPI(title="fabagent-rag", version="0.1.0")

# 上传文件只作为解析过程的临时输入，入库后会删除；真实来源使用用户上传的文件名。
UPLOAD_CACHE_DIR = Path("data/uploads")


class IngestRequest(BaseModel):
    """HTTP 单文件路径入库请求。"""

    path: str = Field(..., description="要入库的单个文件路径")
    batch_size: int = Field(default=10, ge=1, le=100, description="向量化和写入的批大小")


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    inserted: int
    keyword_indexed: int = 0
    sources: list[str] = Field(default_factory=list)


class ParsedUploadDocument(BaseModel):
    source: str
    text: str


class ParseUploadResponse(BaseModel):
    documents: list[ParsedUploadDocument]


class ManualChunkDocument(BaseModel):
    source: str = Field(..., min_length=1, description="chunk 来源文件名")
    chunks: list[str] = Field(..., min_length=1, description="人工确认后的 chunk 文本")


class ChunkConfigRequest(BaseModel):
    chunk_size: int | None = Field(default=None, ge=1, description="chunk 最大字符数")
    chunk_overlap: int | None = Field(default=None, ge=0, description="自动分块重叠字符数")
    min_chunk_size: int | None = Field(default=None, ge=0, description="小 chunk 合并阈值")


class ManualChunkIngestRequest(BaseModel):
    documents: list[ManualChunkDocument] = Field(..., min_length=1)
    chunk_config: ChunkConfigRequest | None = None


class ChunkConfigResponse(BaseModel):
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int


class AskRequest(BaseModel):
    """HTTP 问答请求。"""

    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(default=4, ge=1, le=20, description="检索返回的分块数量")


class AskResponse(BaseModel):
    question: str
    intent: str
    query_plan: dict[str, object] | None = None
    answer: str
    contexts: list[dict[str, object]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/chunk-config", response_model=ChunkConfigResponse)
def chunk_config() -> dict[str, int]:
    """返回前端手动分块使用的默认参数。"""

    config = build_chunk_config(load_settings())
    return {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "min_chunk_size": config.min_chunk_size,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> dict[str, object]:
    """触发文档入库。"""

    path = Path(request.path).expanduser()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"路径不存在：{request.path}")
    if not path.is_file():
        raise HTTPException(status_code=400, detail="该接口只支持单文件路径入库。")

    settings = load_settings()
    result = ingest_path(settings, path, request.batch_size)
    return {**result, "sources": []}


@app.post("/ingest/upload", response_model=IngestResponse)
def ingest_upload(
    files: Annotated[list[UploadFile], File(description="要上传并入库的文档文件")],
) -> dict[str, object]:
    """接收前端上传文件，解析后直接入库。

    FastAPI 的 `UploadFile` 是流式临时文件；MinerU 需要真实文件路径，所以这里先
    写入项目临时目录，并保留文件后缀，保证 PDF/Office/图片能按类型解析。
    """

    settings = load_settings()
    documents = parse_uploaded_files(files, settings)
    result = ingest_documents(settings, documents)
    return {**result, "sources": [document.source for document in documents]}


@app.post("/parse/upload", response_model=ParseUploadResponse)
def parse_upload(
    files: Annotated[list[UploadFile], File(description="要上传并解析预览的文档文件")],
) -> dict[str, object]:
    """只解析上传文件，不写入 Milvus。

    手动分块需要先让前端拿到统一文本，再由用户决定 chunk 边界。
    """

    documents = parse_uploaded_files(files, load_settings())
    return {
        "documents": [
            {"source": document.source, "text": document.text}
            for document in documents
        ]
    }


@app.post("/ingest/chunks", response_model=IngestResponse)
def ingest_chunks(request: ManualChunkIngestRequest) -> dict[str, object]:
    """写入前端人工分好的 chunk。"""

    documents = [(document.source, document.chunks) for document in request.documents]
    settings = load_settings()
    request_config = request.chunk_config
    chunk_config = build_chunk_config(
        settings,
        chunk_size=request_config.chunk_size if request_config else None,
        chunk_overlap=request_config.chunk_overlap if request_config else None,
        min_chunk_size=request_config.min_chunk_size if request_config else None,
    )
    result = ingest_manual_chunks(settings, documents, chunk_config=chunk_config)
    return {**result, "sources": [source for source, _ in documents]}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict[str, object]:
    """查询已入库文档并返回答案和召回上下文。"""

    settings = load_settings()
    return answer_question(settings, request.question, request.top_k)


def parse_uploaded_files(files: list[UploadFile], settings: Settings) -> list[ParsedDocument]:
    """把上传文件保存到临时目录后解析成统一文本。"""

    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个文件。")

    UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    documents: list[ParsedDocument] = []

    with tempfile.TemporaryDirectory(dir=UPLOAD_CACHE_DIR) as temp_dir:
        temp_root = Path(temp_dir)
        for index, upload in enumerate(files):
            source = Path(upload.filename or f"upload-{index}").name
            if not source:
                raise HTTPException(status_code=400, detail="上传文件缺少文件名。")

            temp_path = temp_root / f"{index}{Path(source).suffix.lower()}"
            try:
                with temp_path.open("wb") as target:
                    shutil.copyfileobj(upload.file, target)
                documents.append(
                    parse_document(temp_path, source, mineru_backend=settings.mineru_backend)
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            finally:
                upload.file.close()

    return documents
