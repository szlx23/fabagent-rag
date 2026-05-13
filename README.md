# fabagent-rag

这是一个使用 Milvus 作为向量数据库的轻量 RAG 项目脚手架。

## 包含内容

- 使用 `docker compose` 启动 Milvus standalone 依赖栈
- 默认使用 Milvus `v2.6.15`
- 通过 OpenAI 兼容接口调用嵌入模型
- 提供文档入库和检索问答的 CLI 命令
- 提供 FastAPI 服务接口
- 可选接入 OpenAI 生成最终回答

## 快速开始

1. 创建 Python 虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. 启动 Milvus：

```bash
docker compose up -d
```

3. 复制环境变量模板：

```bash
cp .env.example .env
```

4. 将文档放到 `data/raw` 目录，然后执行入库：

```bash
rag ingest data/raw
```

5. 发起问题：

```bash
rag ask "这个项目是做什么的？"
```

如果没有完整配置推理模型，`rag ask` 会直接返回最相关的检索片段，而不是调用大模型生成回答。

6. 启动 API 服务：

```bash
uvicorn fabagent_rag.api:app --reload
```

启动后访问 `http://127.0.0.1:8000/docs` 查看接口文档。

## 配置

环境变量会从 `.env` 文件中加载。

| 名称 | 默认值 | 说明 |
| --- | --- | --- |
| `MILVUS_HOST` | `localhost` | Milvus 服务地址 |
| `MILVUS_PORT` | `19530` | Milvus gRPC 端口 |
| `MILVUS_COLLECTION` | `rag_documents` | Milvus 集合名称 |
| `EMBEDDING_API_KEY` | 空 | 嵌入模型 API Key；未配置时会读取 `ARK_API_KEY` |
| `EMBEDDING_BASE_URL` | 空 | 嵌入模型 OpenAI 兼容接口地址 |
| `EMBEDDING_MODEL` | `doubao-embedding-text-240715` | 嵌入模型名称 |
| `CHUNK_SIZE` | `800` | 文档分块字符数 |
| `CHUNK_OVERLAP` | `120` | 文档分块重叠字符数 |
| `INFERENCE_API_KEY` | 空 | 推理模型 API Key；未配置时会读取 `ARK_API_KEY` |
| `INFERENCE_BASE_URL` | 空 | 推理模型 OpenAI 兼容接口地址 |
| `INFERENCE_MODEL` | 空 | 推理模型名称 |

## 项目结构

```text
.
+-- docker-compose.yml
+-- pyproject.toml
+-- data
|   +-- raw
+-- scripts
|   +-- reset_milvus.py
+-- src
    +-- fabagent_rag
```

## 常用命令

```bash
rag ingest data/raw --pattern "**/*.md"
rag ask "你的问题" --top-k 5
uvicorn fabagent_rag.api:app --reload
python scripts/reset_milvus.py
```
