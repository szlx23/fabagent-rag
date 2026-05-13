from pathlib import Path
from typing import Annotated
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from fabagent_rag.config import load_settings
from fabagent_rag.documents import load_document_text
from fabagent_rag.rag_service import (
    answer_question,
    ingest_documents,
    ingest_manual_chunks,
    ingest_path,
)

app = FastAPI(title="fabagent-rag", version="0.1.0")

# 上传文件只作为解析过程的临时输入，入库后会删除；真实来源使用用户上传的文件名。
UPLOAD_CACHE_DIR = Path("data/uploads")


class IngestRequest(BaseModel):
    """HTTP 入库请求。

    API 传路径而不是上传文件，是为了先复用本地 CLI 的文件加载能力。
    后续如果需要浏览器上传文件，可以再扩展 multipart endpoint。
    """

    path: str = Field(..., description="要入库的文件或目录路径")
    pattern: str = Field(default="**/*", description="目录检索使用的 glob 模式")
    batch_size: int = Field(default=10, ge=1, le=100, description="向量化和写入的批大小")


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    inserted: int
    sources: list[str] = Field(default_factory=list)


class ParsedUploadDocument(BaseModel):
    source: str
    text: str


class ParseUploadResponse(BaseModel):
    documents: list[ParsedUploadDocument]


class ManualChunkDocument(BaseModel):
    source: str = Field(..., min_length=1, description="chunk 来源文件名")
    chunks: list[str] = Field(..., min_length=1, description="人工确认后的 chunk 文本")


class ManualChunkIngestRequest(BaseModel):
    documents: list[ManualChunkDocument] = Field(..., min_length=1)


class AskRequest(BaseModel):
    """HTTP 问答请求。"""

    question: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(default=4, ge=1, le=20, description="检索返回的分块数量")


class AskResponse(BaseModel):
    question: str
    answer: str
    contexts: list[dict[str, object]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> dict[str, object]:
    """触发文档入库。"""

    path = Path(request.path).expanduser()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"路径不存在：{request.path}")

    settings = load_settings()
    result = ingest_path(settings, path, request.pattern, request.batch_size)
    return {**result, "sources": []}


@app.post("/ingest/upload", response_model=IngestResponse)
def ingest_upload(
    files: Annotated[list[UploadFile], File(description="要上传并入库的文档文件")],
) -> dict[str, object]:
    """接收前端上传文件，解析后直接入库。

    FastAPI 的 `UploadFile` 是流式临时文件；MinerU 需要真实文件路径，所以这里先
    写入项目临时目录，并保留文件后缀，保证 PDF/Office/图片能按类型解析。
    """

    documents = parse_uploaded_files(files)
    settings = load_settings()
    result = ingest_documents(settings, documents)
    return {**result, "sources": [source for source, _ in documents]}


@app.post("/parse/upload", response_model=ParseUploadResponse)
def parse_upload(
    files: Annotated[list[UploadFile], File(description="要上传并解析预览的文档文件")],
) -> dict[str, object]:
    """只解析上传文件，不写入 Milvus。

    手动分块需要先让前端拿到统一文本，再由用户决定 chunk 边界。
    """

    documents = parse_uploaded_files(files)
    return {
        "documents": [
            {"source": source, "text": text}
            for source, text in documents
        ]
    }


@app.post("/ingest/chunks", response_model=IngestResponse)
def ingest_chunks(request: ManualChunkIngestRequest) -> dict[str, object]:
    """写入前端人工分好的 chunk。"""

    documents = [(document.source, document.chunks) for document in request.documents]
    settings = load_settings()
    result = ingest_manual_chunks(settings, documents)
    return {**result, "sources": [source for source, _ in documents]}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict[str, object]:
    """查询已入库文档并返回答案和召回上下文。"""

    settings = load_settings()
    return answer_question(settings, request.question, request.top_k)


def parse_uploaded_files(files: list[UploadFile]) -> list[tuple[str, str]]:
    """把上传文件保存到临时目录后解析成统一文本。"""

    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个文件。")

    UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    documents: list[tuple[str, str]] = []

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
                documents.append((source, load_document_text(temp_path)))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            finally:
                upload.file.close()

    return documents
