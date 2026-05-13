from dataclasses import dataclass
import os
import re

from dotenv import load_dotenv

_MILVUS_COLLECTION_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class Settings:
    milvus_host: str
    milvus_port: str
    milvus_collection: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    inference_api_key: str
    inference_base_url: str
    inference_model: str


def load_settings() -> Settings:
    load_dotenv()
    milvus_collection = os.getenv("MILVUS_COLLECTION", "rag_documents")
    if not _MILVUS_COLLECTION_PATTERN.fullmatch(milvus_collection):
        raise ValueError(
            "MILVUS_COLLECTION 只能包含数字、字母和下划线，"
            f"当前值为 {milvus_collection!r}"
        )

    return Settings(
        milvus_host=os.getenv("MILVUS_HOST", "localhost"),
        milvus_port=os.getenv("MILVUS_PORT", "19530"),
        milvus_collection=milvus_collection,
        embedding_api_key=os.getenv("ARK_API_KEY", ""),
        embedding_base_url=os.getenv("ARK_CODING_PLAN_BASE_URL", ""),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "doubao-embedding-text-240715",
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        inference_api_key=os.getenv("ARK_API_KEY", ""),
        inference_base_url=os.getenv("ARK_CODING_PLAN_BASE_URL", ""),
        inference_model=os.getenv("INFERENCE_MODEL", ""),
    )
