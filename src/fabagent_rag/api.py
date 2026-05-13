from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fabagent_rag.config import load_settings
from fabagent_rag.rag_service import answer_question, ingest_path

app = FastAPI(title="fabagent-rag", version="0.1.0")


class IngestRequest(BaseModel):
    path: str = Field(..., description="要入库的文件或目录路径")
    pattern: str = Field(default="**/*", description="目录检索使用的 glob 模式")
    batch_size: int = Field(default=10, ge=1, le=100, description="向量化和写入的批大小")


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    inserted: int


class AskRequest(BaseModel):
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
def ingest(request: IngestRequest) -> dict[str, int]:
    path = Path(request.path).expanduser()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"路径不存在：{request.path}")

    settings = load_settings()
    return ingest_path(settings, path, request.pattern, request.batch_size)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict[str, object]:
    settings = load_settings()
    return answer_question(settings, request.question, request.top_k)
