import type { AskResponse, IngestResponse } from "../types/rag";

const API_PREFIX = "/api";

async function readJson<T>(response: Response): Promise<T> {
  if (response.ok) {
    return response.json() as Promise<T>;
  }

  // FastAPI 错误通常放在 detail 中。这里集中处理，组件只关心可展示的消息。
  const payload = await response.json().catch(() => null);
  const detail = payload?.detail;
  throw new Error(typeof detail === "string" ? detail : "请求失败，请检查后端服务。");
}

export async function uploadDocuments(files: File[], batchSize: number): Promise<IngestResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append("batch_size", String(batchSize));

  const response = await fetch(`${API_PREFIX}/ingest/upload`, {
    method: "POST",
    body: formData,
  });
  return readJson<IngestResponse>(response);
}

export async function askQuestion(question: string, topK: number): Promise<AskResponse> {
  const response = await fetch(`${API_PREFIX}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      top_k: topK,
    }),
  });
  return readJson<AskResponse>(response);
}
