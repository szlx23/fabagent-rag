import type { ChangeEvent } from "react";
import { useEffect, useMemo, useState } from "react";

import {
  DuplicateSourceError,
  getChunkConfig,
  ingestManualChunks,
  listDocuments,
  parseDocuments,
  uploadDocuments,
} from "../api/rag";
import type {
  ChunkConfig,
  IngestResponse,
  IngestedDocument,
  ParsedUploadDocument,
} from "../types/rag";

const SUPPORTED_EXTENSIONS = new Set([
  ".md",
  ".markdown",
  ".txt",
  ".pdf",
  ".png",
  ".jpg",
  ".jpeg",
  ".docx",
  ".pptx",
  ".xlsx",
  ".html",
  ".htm",
]);

const FALLBACK_CHUNK_CONFIG: ChunkConfig = {
  chunk_size: 800,
  chunk_overlap: 120,
  min_chunk_size: 160,
};

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

type PendingOverwrite =
  | {
      mode: "auto";
      files: File[];
      duplicateSources: string[];
    }
  | {
      mode: "manual";
      documents: ManualDocumentDraft[];
      duplicateSources: string[];
      chunkConfig: ChunkConfig;
    };

function getFileExtension(fileName: string) {
  const lastDot = fileName.lastIndexOf(".");
  return lastDot >= 0 ? fileName.slice(lastDot).toLowerCase() : "";
}

function getFileName(source: string, fileName?: string) {
  const trimmed = (fileName ?? "").trim();
  if (trimmed) {
    return trimmed;
  }
  const fallback = source.split(/[\\/]/).filter(Boolean).pop();
  return fallback || source;
}

function createChunkId(source: string, index: number) {
  return `${source}-${index}-${crypto.randomUUID()}`;
}

function getDisplayNameFromDocument(document: IngestedDocument) {
  return getFileName(document.source, document.file_name);
}

function findRepeatedFileNames(fileNames: string[]) {
  const seen = new Set<string>();
  const duplicates: string[] = [];

  for (const fileName of fileNames) {
    if (seen.has(fileName) && !duplicates.includes(fileName)) {
      duplicates.push(fileName);
    }
    seen.add(fileName);
  }

  return duplicates;
}

function findDuplicateFileNames(existing: IngestedDocument[], fileNames: string[]) {
  const existingNames = new Set(existing.map((document) => getDisplayNameFromDocument(document)));
  const duplicates: string[] = [];

  for (const fileName of fileNames) {
    if (existingNames.has(fileName) && !duplicates.includes(fileName)) {
      duplicates.push(fileName);
    }
  }

  return duplicates;
}

function createInitialChunks(document: ParsedUploadDocument, config: ChunkConfig): ChunkDraft[] {
  // 手动分块的初始稿会按用户配置做预切分；用户仍可以继续编辑边界。
  const sections = document.text
    .split(/\n{2,}/)
    .map((section) => section.trim())
    .filter(Boolean);

  if (sections.length === 0) {
    return [{ id: createChunkId(document.source, 0), text: document.text }];
  }

  const packedSections = packSectionsByChunkSize(sections, config.chunk_size);
  const mergedSections = mergeSmallSections(packedSections, config);
  return mergedSections.map((section, index) => ({
    id: createChunkId(document.source, index),
    text: section,
  }));
}

function packSectionsByChunkSize(sections: string[], chunkSize: number) {
  const chunks: string[] = [];
  let current = "";

  for (const section of sections) {
    const next = current ? `${current}\n\n${section}` : section;
    if (next.length <= chunkSize || !current) {
      current = next;
      continue;
    }
    chunks.push(current);
    current = section;
  }

  if (current) {
    chunks.push(current);
  }
  return chunks;
}

function mergeSmallSections(sections: string[], config: ChunkConfig) {
  const chunks: string[] = [];
  let index = 0;

  while (index < sections.length) {
    const current = sections[index];
    if (current.length >= config.min_chunk_size) {
      chunks.push(current);
      index += 1;
      continue;
    }

    if (chunks.length > 0 && joinChunkText(chunks[chunks.length - 1], current).length <= config.chunk_size) {
      chunks[chunks.length - 1] = joinChunkText(chunks[chunks.length - 1], current);
      index += 1;
      continue;
    }

    const next = sections[index + 1];
    if (next && joinChunkText(current, next).length <= config.chunk_size) {
      chunks.push(joinChunkText(current, next));
      index += 2;
      continue;
    }

    chunks.push(current);
    index += 1;
  }

  return chunks;
}

function joinChunkText(left: string, right: string) {
  return `${left.trim()}\n\n${right.trim()}`.trim();
}

export function FileUploadPanel({ onIngested }: FileUploadPanelProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [chunkMode, setChunkMode] = useState<"auto" | "manual">("auto");
  const [chunkConfig, setChunkConfig] = useState<ChunkConfig>(FALLBACK_CHUNK_CONFIG);
  const [manualDocuments, setManualDocuments] = useState<ManualDocumentDraft[]>([]);
  const [selectedDocumentIndex, setSelectedDocumentIndex] = useState(0);
  const [pendingOverwrite, setPendingOverwrite] = useState<PendingOverwrite | null>(null);
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
  const chunkConfigValid =
    chunkConfig.chunk_size > 0 &&
    chunkConfig.chunk_overlap >= 0 &&
    chunkConfig.min_chunk_size >= 0 &&
    chunkConfig.chunk_overlap < chunkConfig.chunk_size &&
    chunkConfig.min_chunk_size <= chunkConfig.chunk_size;

  useEffect(() => {
    getChunkConfig()
      .then(setChunkConfig)
      .catch(() => {
        setChunkConfig(FALLBACK_CHUNK_CONFIG);
      });
  }, []);

  function handleFilesSelected(event: ChangeEvent<HTMLInputElement>) {
    const selectedFiles = Array.from(event.target.files ?? []);
    setFiles((currentFiles) => [...currentFiles, ...selectedFiles]);
    setManualDocuments([]);
    setSelectedDocumentIndex(0);
    setPendingOverwrite(null);
    setStatus("idle");
    setMessage("");

    // 清空 input 的值，用户删除后可以重新选择同一个文件。
    event.target.value = "";
  }

  function removeFile(fileIndex: number) {
    setFiles((currentFiles) => currentFiles.filter((_, index) => index !== fileIndex));
    setManualDocuments([]);
    setSelectedDocumentIndex(0);
    setPendingOverwrite(null);
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

  function updateChunkConfig(field: keyof ChunkConfig, value: number) {
    setChunkConfig((currentConfig) => ({
      ...currentConfig,
      [field]: Number.isFinite(value) ? value : currentConfig[field],
    }));
  }

  function regenerateManualChunks() {
    setManualDocuments((currentDocuments) =>
      currentDocuments.map((document) => ({
        ...document,
        chunks: createInitialChunks(
          {
            source: document.source,
            text: document.chunks.map((chunk) => chunk.text).join("\n\n"),
          },
          chunkConfig,
        ),
      })),
    );
  }

  async function getDuplicateSources(fileNames: string[]) {
    const response = await listDocuments();
    const duplicates = findDuplicateFileNames(response.documents, fileNames);
    return duplicates;
  }

  async function handleAutoUpload(overwriteExisting = false, overrideFiles?: File[]) {
    const targetFiles = overrideFiles ?? files;
    if (!canStart || !chunkConfigValid) {
      setStatus("error");
      setMessage(
        !chunkConfigValid
          ? "请检查 chunk 参数：overlap 和小 chunk 阈值必须小于 chunk 上限。"
          : targetFiles.length === 0
            ? "请选择至少一个文件。"
            : "请先删除不支持的文件。",
      );
      return;
    }

    if (!overwriteExisting) {
      try {
        const repeatedSources = findRepeatedFileNames(targetFiles.map((file) => file.name));
        if (repeatedSources.length > 0) {
          setStatus("error");
          setMessage(`本次选择中存在重复文件名：${repeatedSources.join("、")}。请先去重。`);
          return;
        }

        const duplicateSources = await getDuplicateSources(targetFiles.map((file) => file.name));
        if (duplicateSources.length > 0) {
          setPendingOverwrite({
            mode: "auto",
            files: targetFiles,
            duplicateSources,
          });
          setStatus("idle");
          setMessage(`检测到重复文件名：${duplicateSources.join("、")}。是否覆盖后重新入库？`);
          return;
        }
      } catch (error) {
        setStatus("error");
        setMessage(error instanceof Error ? error.message : "重复文件检查失败。");
        return;
      }
    }

    setPendingOverwrite(null);
    setStatus("uploading");
    setMessage(overwriteExisting ? "正在覆盖并重新入库..." : "正在自动解析、分块并写入向量库...");

    try {
      const result = await uploadDocuments(targetFiles, overwriteExisting);
      setStatus("done");
      setMessage(`已写入 ${result.inserted} 个分块，来源 ${result.documents} 个文件。`);
      onIngested(result);
    } catch (error) {
      if (error instanceof DuplicateSourceError) {
        setPendingOverwrite({
          mode: "auto",
          files: targetFiles,
          duplicateSources: error.duplicateSources,
        });
        setStatus("idle");
        setMessage(`检测到重复文件名：${error.duplicateSources.join("、")}。是否覆盖后重新入库？`);
        return;
      }
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "上传失败。");
    }
  }

  async function handleManualIngest(
    overwriteExisting = false,
    overrideDocuments?: ManualDocumentDraft[],
    overrideChunkConfig?: ChunkConfig,
  ) {
    const targetDocuments = overrideDocuments ?? manualDocuments;
    const targetChunkConfig = overrideChunkConfig ?? chunkConfig;
    const documents = targetDocuments.map((document) => ({
      source: document.source,
      chunks: document.chunks.map((chunk) => chunk.text).filter((text) => text.trim()),
    }));

    if (documents.every((document) => document.chunks.length === 0)) {
      setStatus("error");
      setMessage("请至少保留一个非空 chunk。");
      return;
    }

    if (!overwriteExisting) {
      try {
        const repeatedSources = findRepeatedFileNames(
          targetDocuments.map((document) => document.source),
        );
        if (repeatedSources.length > 0) {
          setStatus("error");
          setMessage(`本次提交中存在重复文件名：${repeatedSources.join("、")}。请先去重。`);
          return;
        }

        const duplicateSources = await getDuplicateSources(targetDocuments.map((document) => document.source));
        if (duplicateSources.length > 0) {
          setPendingOverwrite({
            mode: "manual",
            documents: targetDocuments,
            duplicateSources,
            chunkConfig: targetChunkConfig,
          });
          setStatus("idle");
          setMessage(`检测到重复文件名：${duplicateSources.join("、")}。是否覆盖后重新入库？`);
          return;
        }
      } catch (error) {
        setStatus("error");
        setMessage(error instanceof Error ? error.message : "重复文件检查失败。");
        return;
      }
    }

    setPendingOverwrite(null);
    setStatus("uploading");
    setMessage(overwriteExisting ? "正在覆盖并写入手动确认的 chunk..." : "正在写入手动确认的 chunk...");

    try {
      const result = await ingestManualChunks(documents, targetChunkConfig, overwriteExisting);
      setStatus("done");
      setMessage(`已写入 ${result.inserted} 个手动 chunk。`);
      onIngested(result);
    } catch (error) {
      if (error instanceof DuplicateSourceError) {
        setPendingOverwrite({
          mode: "manual",
          documents: targetDocuments,
          duplicateSources: error.duplicateSources,
          chunkConfig: targetChunkConfig,
        });
        setStatus("idle");
        setMessage(`检测到重复文件名：${error.duplicateSources.join("、")}。是否覆盖后重新入库？`);
        return;
      }
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "入库失败。");
    }
  }

  function cancelOverwrite() {
    setPendingOverwrite(null);
    setStatus("idle");
    setMessage("已取消覆盖操作。");
  }

  function confirmOverwrite() {
    if (!pendingOverwrite) {
      return;
    }

    if (pendingOverwrite.mode === "auto") {
      void handleAutoUpload(true, pendingOverwrite.files);
      return;
    }

    void handleManualIngest(true, pendingOverwrite.documents, pendingOverwrite.chunkConfig);
  }

  async function handleParseForManualChunks() {
    setPendingOverwrite(null);
    if (!canStart || !chunkConfigValid) {
      setStatus("error");
      setMessage(
        !chunkConfigValid
          ? "请检查 chunk 参数：overlap 和小 chunk 阈值必须小于 chunk 上限。"
          : files.length === 0
            ? "请选择至少一个文件。"
            : "请先删除不支持的文件。",
      );
      return;
    }

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
          chunks: createInitialChunks(document, chunkConfig),
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

  return (
    <section className="panel ingestionPanel">
      <div className="panelHeader">
        <div>
          <h2>文档入库</h2>
        </div>
        <span className="panelBadge">{chunkMode === "auto" ? "自动分块" : "手动分块"}</span>
      </div>

      <div className="processRail" aria-label="入库流程">
        <span className={files.length > 0 ? "done" : "active"}>选择文件</span>
        <span className={chunkMode === "manual" && manualDocuments.length > 0 ? "done" : ""}>
          解析预览
        </span>
        <span className={status === "done" ? "done" : ""}>写入索引</span>
      </div>

      <div className="segmentedControl" aria-label="分块模式">
        <button
          className={chunkMode === "auto" ? "active" : ""}
          disabled={busy}
          onClick={() => {
            setPendingOverwrite(null);
            setChunkMode("auto");
          }}
          type="button"
        >
          自动分块
        </button>
        <button
          className={chunkMode === "manual" ? "active" : ""}
          disabled={busy}
          onClick={() => {
            setPendingOverwrite(null);
            setChunkMode("manual");
          }}
          type="button"
        >
          手动分块
        </button>
      </div>

      <label className="dropZone">
        <input
          type="file"
          multiple
          accept=".md,.markdown,.txt,.pdf,.png,.jpg,.jpeg,.docx,.pptx,.xlsx,.html,.htm"
          onChange={handleFilesSelected}
        />
        <span className="dropIcon" aria-hidden="true">
          +
        </span>
        <span className="dropCopy">
          <strong>{files.length > 0 ? `${files.length} 个文件待处理` : "拖入或选择文档"}</strong>
          <small>PDF、Office、表格、网页、文本、图片</small>
        </span>
        <span className="dropAction">选择</span>
      </label>

      {files.length > 0 && (
        <div className="fileList">
          <div className="fileListHeader">
            <strong>待处理文件</strong>
            <span>{files.length} 个，{(totalSize / 1024).toFixed(1)} KB</span>
          </div>
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
        </div>
      )}

      {chunkMode === "auto" ? (
        <div className="actionBar">
          <span className="modeHint">自动解析、自动分块，适合批量导入测试资料。</span>
          <button disabled={!canStart} onClick={() => void handleAutoUpload()} type="button">
            {status === "uploading" ? "处理中" : "解析并入库"}
          </button>
        </div>
      ) : (
        <div className="manualChunkFlow">
          <div className="chunkConfigPanel">
            <label>
              Chunk 上限
              <input
                min={100}
                step={50}
                type="number"
                value={chunkConfig.chunk_size}
                onChange={(event) => updateChunkConfig("chunk_size", Number(event.target.value))}
              />
            </label>
            <label>
              Overlap
              <input
                min={0}
                step={20}
                type="number"
                value={chunkConfig.chunk_overlap}
                onChange={(event) => updateChunkConfig("chunk_overlap", Number(event.target.value))}
              />
            </label>
            <label>
              小 chunk 阈值
              <input
                min={0}
                step={20}
                type="number"
                value={chunkConfig.min_chunk_size}
                onChange={(event) => updateChunkConfig("min_chunk_size", Number(event.target.value))}
              />
            </label>
            <button
              className="secondaryButton"
              disabled={busy || manualDocuments.length === 0 || !chunkConfigValid}
              onClick={regenerateManualChunks}
              type="button"
            >
              按配置重分
            </button>
          </div>
          {!chunkConfigValid && (
            <p className="statusText error">
              参数无效：Overlap 和小 chunk 阈值不能大于或等于 chunk 上限。
            </p>
          )}
          <div className="actionBar">
            <span className="modeHint">解析后编辑分块</span>
            <button
              disabled={!canStart || !chunkConfigValid}
              onClick={handleParseForManualChunks}
              type="button"
            >
              {status === "parsing" ? "解析中" : "解析预览"}
            </button>
          </div>

          {manualDocuments.length > 0 && (
            <div className="chunkEditor">
              <div className="chunkSummary">
                <div>
                <strong>{manualChunkCount} 个分块</strong>
                <span>{manualDocuments.length} 个文档已解析</span>
              </div>
                <button disabled={busy} onClick={() => void handleManualIngest()} type="button">
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
                        <strong>分块 {index + 1}</strong>
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

      {pendingOverwrite && (
        <div className="statusText error" role="alert">
          <strong>发现重复文件名：</strong>
          {pendingOverwrite.duplicateSources.join("、")}
          <div className="inlineActions">
            <button className="secondaryButton" disabled={busy} onClick={cancelOverwrite} type="button">
              取消覆盖
            </button>
            <button disabled={busy} onClick={confirmOverwrite} type="button">
              覆盖并重新入库
            </button>
          </div>
        </div>
      )}

      {unsupportedFiles.length > 0 && (
        <p className="statusText error">存在不支持的文件，请删除后再上传。</p>
      )}
      {message && <p className={`statusText ${status}`}>{message}</p>}
    </section>
  );
}
