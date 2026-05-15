import type {
  AskResponse,
  ChunkConfig,
  DeleteDocumentsResponse,
  DocumentsResponse,
  IngestResponse,
  ParseUploadResponse,
} from "../types/rag";

const API_PREFIX = "/api";

export class DuplicateSourceError extends Error {
  duplicateSources: string[];

  constructor(message: string, duplicateSources: string[]) {
    super(message);
    this.name = "DuplicateSourceError";
    this.duplicateSources = duplicateSources;
  }
}

async function readJson<T>(response: Response): Promise<T> {
  if (response.ok) {
    return response.json() as Promise<T>;
  }

  // FastAPI 错误通常放在 detail 中。这里集中处理，组件只关心可展示的消息。
  const payload = await response.json().catch(() => null);
  const detail = payload?.detail;
  if (detail && typeof detail === "object") {
    const duplicateSources = detail.duplicate_sources;
    if (Array.isArray(duplicateSources) && response.status === 409) {
      throw new DuplicateSourceError(
        detail.message ?? "检测到重复文件名，确认覆盖后再重新入库。",
        duplicateSources.map((source) => String(source)),
      );
    }
    if (typeof detail.message === "string") {
      throw new Error(detail.message);
    }
  }
  throw new Error(typeof detail === "string" ? detail : "请求失败，请检查后端服务。");
}

export async function uploadDocuments(
  files: File[],
  overwriteExisting = false,
): Promise<IngestResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append("overwrite_existing", String(overwriteExisting));

  const response = await fetch(`${API_PREFIX}/ingest/upload`, {
    method: "POST",
    body: formData,
  });
  return readJson<IngestResponse>(response);
}

export async function parseDocuments(files: File[]): Promise<ParseUploadResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  const response = await fetch(`${API_PREFIX}/parse/upload`, {
    method: "POST",
    body: formData,
  });
  return readJson<ParseUploadResponse>(response);
}

export async function ingestManualChunks(
  documents: Array<{ source: string; chunks: string[] }>,
  chunkConfig: ChunkConfig,
  overwriteExisting = false,
): Promise<IngestResponse> {
  const response = await fetch(`${API_PREFIX}/ingest/chunks`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      documents,
      chunk_config: chunkConfig,
      overwrite_existing: overwriteExisting,
    }),
  });
  return readJson<IngestResponse>(response);
}

export async function getChunkConfig(): Promise<ChunkConfig> {
  const response = await fetch(`${API_PREFIX}/chunk-config`);
  return readJson<ChunkConfig>(response);
}

export async function listDocuments(): Promise<DocumentsResponse> {
  const response = await fetch(`${API_PREFIX}/documents`);
  return readJson<DocumentsResponse>(response);
}

export async function deleteDocuments(sources: string[]): Promise<DeleteDocumentsResponse> {
  const response = await fetch(`${API_PREFIX}/documents`, {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ sources }),
  });
  return readJson<DeleteDocumentsResponse>(response);
}

export async function askQuestion(
  question: string,
  topK: number,
  selectedSources: string[],
): Promise<AskResponse> {
  const response = await fetch(`${API_PREFIX}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      top_k: topK,
      selected_sources: selectedSources,
    }),
  });
  return readJson<AskResponse>(response);
}
