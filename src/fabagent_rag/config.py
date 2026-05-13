from dataclasses import dataclass
import os
import re

from dotenv import load_dotenv

_MILVUS_COLLECTION_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class Settings:
    """运行时配置的快照。

    项目所有入口（CLI 和 FastAPI）都会先调用 `load_settings()`，再把这个对象
    传给服务层。这样可以避免业务代码到处直接读取环境变量，review 时只需要从
    这里理解“系统有哪些外部依赖”。
    """

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
    """从 `.env` 和系统环境变量加载配置。

    注意：当前项目把豆包/火山方舟的 API key 同时用于 embedding 和推理。
    如果后续要拆分成不同账号或不同网关，只需要扩展这里的字段读取逻辑。
    """

    load_dotenv()
    milvus_collection = os.getenv("MILVUS_COLLECTION", "rag_documents")
    if not _MILVUS_COLLECTION_PATTERN.fullmatch(milvus_collection):
        # Milvus collection 名称不能包含短横线等字符。提前校验可以避免
        # PyMilvus 抛出很长的底层异常，定位起来更直接。
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
