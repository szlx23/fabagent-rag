import type { ChangeEvent } from "react";
import { useMemo, useState } from "react";

import { ingestManualChunks, parseDocuments, uploadDocuments } from "../api/rag";
import type { IngestResponse, ParsedUploadDocument } from "../types/rag";

const SUPPORTED_EXTENSIONS = new Set([
  ".md",
  ".markdown",
  ".txt",
  ".pdf",
  ".png",
  ".jpg",
  ".jpeg",
  ".doc",
  ".docx",
  ".ppt",
  ".pptx",
  ".xlsx",
  ".html",
  ".htm",
]);

type FileUploadPanelProps = {
  onIngested: (result: IngestResponse) => void;
};

type ChunkDraft = {
  id: string;
  text: string;
};

type ManualDocumentDraft = {
  source: string;
  chunks: ChunkDraft[];
};

function getFileExtension(fileName: string) {
  const lastDot = fileName.lastIndexOf(".");
  return lastDot >= 0 ? fileName.slice(lastDot).toLowerCase() : "";
}

function createChunkId(source: string, index: number) {
  return `${source}-${index}-${crypto.randomUUID()}`;
}

function createInitialChunks(document: ParsedUploadDocument): ChunkDraft[] {
  // 手动分块的初始稿只做温和预切分：按 Markdown/文本段落分开，用户再调整边界。
  const sections = document.text
    .split(/\n{2,}/)
    .map((section) => section.trim())
    .filter(Boolean);

  if (sections.length === 0) {
    return [{ id: createChunkId(document.source, 0), text: document.text }];
  }

  return sections.map((section, index) => ({
    id: createChunkId(document.source, index),
    text: section,
  }));
}

export function FileUploadPanel({ onIngested }: FileUploadPanelProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [chunkMode, setChunkMode] = useState<"auto" | "manual">("auto");
  const [manualDocuments, setManualDocuments] = useState<ManualDocumentDraft[]>([]);
  const [selectedDocumentIndex, setSelectedDocumentIndex] = useState(0);
  const [status, setStatus] = useState<"idle" | "parsing" | "uploading" | "done" | "error">(
    "idle",
  );
  const [message, setMessage] = useState("");

  const unsupportedFiles = useMemo(
    () => files.filter((file) => !SUPPORTED_EXTENSIONS.has(getFileExtension(file.name))),
    [files],
  );
  const totalSize = useMemo(
    () => files.reduce((total, file) => total + file.size, 0),
    [files],
  );
  const selectedDocument = manualDocuments[selectedDocumentIndex];
  const manualChunkCount = manualDocuments.reduce(
    (total, document) => total + document.chunks.filter((chunk) => chunk.text.trim()).length,
    0,
  );
  const busy = status === "parsing" || status === "uploading";
  const canStart = files.length > 0 && unsupportedFiles.length === 0 && !busy;

  function handleFilesSelected(event: ChangeEvent<HTMLInputElement>) {
    const selectedFiles = Array.from(event.target.files ?? []);
    setFiles((currentFiles) => [...currentFiles, ...selectedFiles]);
    setManualDocuments([]);
    setSelectedDocumentIndex(0);
    setStatus("idle");
    setMessage("");

    // 清空 input 的值，用户删除后可以重新选择同一个文件。
    event.target.value = "";
  }

  function removeFile(fileIndex: number) {
    setFiles((currentFiles) => currentFiles.filter((_, index) => index !== fileIndex));
    setManualDocuments([]);
    setSelectedDocumentIndex(0);
    setStatus("idle");
    setMessage("");
  }

  function updateChunk(documentIndex: number, chunkId: string, text: string) {
    setManualDocuments((currentDocuments) =>
      currentDocuments.map((document, index) =>
        index === documentIndex
          ? {
              ...document,
              chunks: document.chunks.map((chunk) =>
                chunk.id === chunkId ? { ...chunk, text } : chunk,
              ),
            }
          : document,
      ),
    );
  }

  function addChunk(documentIndex: number, afterIndex?: number) {
    setManualDocuments((currentDocuments) =>
      currentDocuments.map((document, index) => {
        if (index !== documentIndex) {
          return document;
        }

        const nextChunk = { id: createChunkId(document.source, document.chunks.length), text: "" };
        const nextChunks = [...document.chunks];
        nextChunks.splice(afterIndex === undefined ? nextChunks.length : afterIndex + 1, 0, nextChunk);
        return { ...document, chunks: nextChunks };
      }),
    );
  }

  function removeChunk(documentIndex: number, chunkId: string) {
    setManualDocuments((currentDocuments) =>
      currentDocuments.map((document, index) =>
        index === documentIndex
          ? {
              ...document,
              chunks:
                document.chunks.length > 1
                  ? document.chunks.filter((chunk) => chunk.id !== chunkId)
                  : document.chunks,
            }
          : document,
      ),
    );
  }

  async function handleAutoUpload() {
    if (!canStart) {
      setStatus("error");
      setMessage(files.length === 0 ? "请选择至少一个文件。" : "请先删除不支持的文件。");
      return;
    }

    setStatus("uploading");
    setMessage("正在自动解析、分块并写入向量库...");

    try {
      const result = await uploadDocuments(files);
      setStatus("done");
      setMessage(`已写入 ${result.inserted} 个分块，来源 ${result.documents} 个文件。`);
      onIngested(result);
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "上传失败。");
    }
  }

  async function handleParseForManualChunks() {
    if (!canStart) {
      setStatus("error");
      setMessage(files.length === 0 ? "请选择至少一个文件。" : "请先删除不支持的文件。");
      return;
    }

    setStatus("parsing");
    setMessage("正在解析文档，稍后可手动调整 chunk...");

    try {
      const result = await parseDocuments(files);
      setManualDocuments(
        result.documents.map((document) => ({
          source: document.source,
          chunks: createInitialChunks(document),
        })),
      );
      setSelectedDocumentIndex(0);
      setStatus("idle");
      setMessage("解析完成，请检查并调整 chunk 后入库。");
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "解析失败。");
    }
  }

  async function handleManualIngest() {
    const documents = manualDocuments.map((document) => ({
      source: document.source,
      chunks: document.chunks.map((chunk) => chunk.text).filter((text) => text.trim()),
    }));

    if (documents.every((document) => document.chunks.length === 0)) {
      setStatus("error");
      setMessage("请至少保留一个非空 chunk。");
      return;
    }

    setStatus("uploading");
    setMessage("正在写入手动确认的 chunk...");

    try {
      const result = await ingestManualChunks(documents);
      setStatus("done");
      setMessage(`已写入 ${result.inserted} 个手动 chunk。`);
      onIngested(result);
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "入库失败。");
    }
  }

  return (
    <section className="panel">
      <div className="panelHeader">
        <div>
          <span className="panelLabel">Ingestion</span>
          <h2>文档入库</h2>
          <p>自动分块适合批量测试；手动分块适合检查解析质量和边界。</p>
        </div>
      </div>

      <div className="segmentedControl" aria-label="分块模式">
        <button
          className={chunkMode === "auto" ? "active" : ""}
          disabled={busy}
          onClick={() => setChunkMode("auto")}
          type="button"
        >
          自动分块
        </button>
        <button
          className={chunkMode === "manual" ? "active" : ""}
          disabled={busy}
          onClick={() => setChunkMode("manual")}
          type="button"
        >
          手动分块
        </button>
      </div>

      <label className="dropZone">
        <input
          type="file"
          multiple
          accept=".md,.markdown,.txt,.pdf,.png,.jpg,.jpeg,.doc,.docx,.ppt,.pptx,.xlsx,.html,.htm"
          onChange={handleFilesSelected}
        />
        <span className="dropAction">选择文件</span>
        <strong>{files.length > 0 ? `${files.length} 个文件待处理` : "拖入或选择文档"}</strong>
        <small>PDF / DOC / DOCX / PPT / PPTX / XLSX / HTML / MD / TXT / 图片</small>
      </label>

      {files.length > 0 && (
        <div className="fileList">
          {files.map((file, index) => (
            <div className="fileRow" key={`${file.name}-${file.size}`}>
              <div className="fileNameGroup">
                <span>{file.name}</span>
                {!SUPPORTED_EXTENSIONS.has(getFileExtension(file.name)) && (
                  <small className="fileWarning">不支持</small>
                )}
              </div>
              <div className="fileActions">
                <small>{(file.size / 1024).toFixed(1)} KB</small>
                <button
                  aria-label={`删除 ${file.name}`}
                  className="iconButton"
                  disabled={busy}
                  onClick={() => removeFile(index)}
                  type="button"
                >
                  删除
                </button>
              </div>
            </div>
          ))}
          <div className="fileTotal">合计 {(totalSize / 1024).toFixed(1)} KB</div>
        </div>
      )}

      {chunkMode === "auto" ? (
        <div className="controlRow">
          <span className="modeHint">使用后端默认 chunk 策略。</span>
          <button disabled={!canStart} onClick={handleAutoUpload} type="button">
            {status === "uploading" ? "处理中" : "解析并入库"}
          </button>
        </div>
      ) : (
        <div className="manualChunkFlow">
          <div className="controlRow">
            <span className="modeHint">先解析预览，再手动编辑 chunk。</span>
            <button disabled={!canStart} onClick={handleParseForManualChunks} type="button">
              {status === "parsing" ? "解析中" : "解析预览"}
            </button>
          </div>

          {manualDocuments.length > 0 && (
            <div className="chunkEditor">
              <div className="chunkSummary">
                <strong>{manualChunkCount} 个 chunk</strong>
                <button disabled={busy} onClick={handleManualIngest} type="button">
                  {status === "uploading" ? "写入中" : "确认入库"}
                </button>
              </div>

              <div className="documentTabs">
                {manualDocuments.map((document, index) => (
                  <button
                    className={index === selectedDocumentIndex ? "active" : ""}
                    disabled={busy}
                    key={document.source}
                    onClick={() => setSelectedDocumentIndex(index)}
                    type="button"
                  >
                    {document.source}
                  </button>
                ))}
              </div>

              {selectedDocument && (
                <div className="chunkList">
                  {selectedDocument.chunks.map((chunk, index) => (
                    <article className="chunkCard" key={chunk.id}>
                      <div className="chunkCardHeader">
                        <strong>Chunk {index + 1}</strong>
                        <span>{chunk.text.trim().length} 字符</span>
                      </div>
                      <textarea
                        disabled={busy}
                        onChange={(event) =>
                          updateChunk(selectedDocumentIndex, chunk.id, event.target.value)
                        }
                        rows={6}
                        value={chunk.text}
                      />
                      <div className="chunkActions">
                        <button
                          className="secondaryButton"
                          disabled={busy}
                          onClick={() => addChunk(selectedDocumentIndex, index)}
                          type="button"
                        >
                          插入 chunk
                        </button>
                        <button
                          className="iconButton"
                          disabled={busy || selectedDocument.chunks.length === 1}
                          onClick={() => removeChunk(selectedDocumentIndex, chunk.id)}
                          type="button"
                        >
                          删除
                        </button>
                      </div>
                    </article>
                  ))}
                  <button
                    className="secondaryButton fullWidth"
                    disabled={busy}
                    onClick={() => addChunk(selectedDocumentIndex)}
                    type="button"
                  >
                    添加空 chunk
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {unsupportedFiles.length > 0 && (
        <p className="statusText error">存在不支持的文件，请删除后再上传。</p>
      )}
      {message && <p className={`statusText ${status}`}>{message}</p>}
    </section>
  );
}
