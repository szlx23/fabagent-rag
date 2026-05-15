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

type PendingAction =
  | { kind: "auto"; files: File[]; duplicateSources: string[] }
  | { kind: "parse"; files: File[]; duplicateSources: string[] }
  | {
      kind: "manual";
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
  return source.split(/[\\/]/).filter(Boolean).pop() || source;
}

function createChunkId(source: string, index: number) {
  return `${source}-${index}-${crypto.randomUUID()}`;
}

function getDisplayNameFromDocument(document: IngestedDocument) {
  return getFileName(document.source, document.file_name);
}

function uniqueDuplicates(names: string[]) {
  const seen = new Set<string>();
  const duplicates: string[] = [];
  for (const name of names) {
    if (seen.has(name) && !duplicates.includes(name)) {
      duplicates.push(name);
    }
    seen.add(name);
  }
  return duplicates;
}

function findExistingDuplicates(existing: IngestedDocument[], names: string[]) {
  const existingNames = new Set(existing.map((document) => getDisplayNameFromDocument(document)));
  return names.filter((name, index) => existingNames.has(name) && names.indexOf(name) === index);
}

function createInitialChunks(document: ParsedUploadDocument, config: ChunkConfig): ChunkDraft[] {
  const sections = document.text
    .split(/\n{2,}/)
    .map((section) => section.trim())
    .filter(Boolean);

  if (sections.length === 0) {
    return [{ id: createChunkId(document.source, 0), text: document.text }];
  }

  const packed = packSectionsByChunkSize(sections, config.chunk_size);
  const merged = mergeSmallSections(packed, config);
  return merged.map((section, index) => ({
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
    const previous = chunks[chunks.length - 1];
    if (previous && `${previous}\n\n${current}`.length <= config.chunk_size) {
      chunks[chunks.length - 1] = `${previous}\n\n${current}`.trim();
      index += 1;
      continue;
    }
    const next = sections[index + 1];
    if (next && `${current}\n\n${next}`.length <= config.chunk_size) {
      chunks.push(`${current}\n\n${next}`.trim());
      index += 2;
      continue;
    }
    chunks.push(current);
    index += 1;
  }
  return chunks;
}

export function FileUploadPanel({ onIngested }: FileUploadPanelProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [mode, setMode] = useState<"auto" | "manual">("auto");
  const [chunkConfig, setChunkConfig] = useState<ChunkConfig>(FALLBACK_CHUNK_CONFIG);
  const [manualDocuments, setManualDocuments] = useState<ManualDocumentDraft[]>([]);
  const [selectedDocumentIndex, setSelectedDocumentIndex] = useState(0);
  const [approvedOverwriteSources, setApprovedOverwriteSources] = useState<string[]>([]);
  const [pendingAction, setPendingAction] = useState<PendingAction | null>(null);
  const [status, setStatus] = useState<"idle" | "parsing" | "uploading" | "done" | "error">(
    "idle",
  );
  const [message, setMessage] = useState("");

  const unsupportedFiles = useMemo(
    () => files.filter((file) => !SUPPORTED_EXTENSIONS.has(getFileExtension(file.name))),
    [files],
  );
  const totalSize = useMemo(() => files.reduce((total, file) => total + file.size, 0), [files]);
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
      .catch(() => setChunkConfig(FALLBACK_CHUNK_CONFIG));
  }, []);

  function resetTransientState() {
    setPendingAction(null);
    setApprovedOverwriteSources([]);
    setMessage("");
    setStatus("idle");
  }

  function handleFilesSelected(event: ChangeEvent<HTMLInputElement>) {
    const selectedFiles = Array.from(event.target.files ?? []);
    setFiles((currentFiles) => [...currentFiles, ...selectedFiles]);
    setManualDocuments([]);
    setSelectedDocumentIndex(0);
    resetTransientState();
    event.target.value = "";
  }

  function removeFile(fileIndex: number) {
    setFiles((currentFiles) => currentFiles.filter((_, index) => index !== fileIndex));
    setManualDocuments([]);
    setSelectedDocumentIndex(0);
    resetTransientState();
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

  async function getExistingDuplicates(names: string[]) {
    const repeated = uniqueDuplicates(names);
    if (repeated.length > 0) {
      throw new Error(`本次选择中存在重复文件名：${repeated.join("、")}`);
    }
    const response = await listDocuments();
    return findExistingDuplicates(response.documents, names);
  }

  async function runAutoUpload(overwriteExisting = false, targetFiles = files) {
    if (!canStart || !chunkConfigValid) {
      setStatus("error");
      setMessage(!chunkConfigValid ? "Chunk 参数无效。" : "请选择可入库文件。");
      return;
    }

    if (!overwriteExisting) {
      try {
        const duplicates = await getExistingDuplicates(targetFiles.map((file) => file.name));
        if (duplicates.length > 0) {
          setPendingAction({ kind: "auto", files: targetFiles, duplicateSources: duplicates });
          setMessage("");
          setStatus("idle");
          return;
        }
      } catch (error) {
        setStatus("error");
        setMessage(error instanceof Error ? error.message : "重复文件检查失败。");
        return;
      }
    }

    setPendingAction(null);
    setStatus("uploading");
    setMessage(overwriteExisting ? "覆盖写入中..." : "写入中...");
    try {
      const result = await uploadDocuments(targetFiles, overwriteExisting);
      setStatus("done");
      setMessage(`写入 ${result.inserted} 个分块。`);
      onIngested(result);
    } catch (error) {
      if (error instanceof DuplicateSourceError) {
        setPendingAction({
          kind: "auto",
          files: targetFiles,
          duplicateSources: error.duplicateSources,
        });
        setStatus("idle");
        setMessage("");
        return;
      }
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "入库失败。");
    }
  }

  async function runParse(overwriteApproved = false, targetFiles = files) {
    if (!canStart || !chunkConfigValid) {
      setStatus("error");
      setMessage(!chunkConfigValid ? "Chunk 参数无效。" : "请选择可解析文件。");
      return;
    }

    if (!overwriteApproved) {
      try {
        const duplicates = await getExistingDuplicates(targetFiles.map((file) => file.name));
        if (duplicates.length > 0) {
          setPendingAction({ kind: "parse", files: targetFiles, duplicateSources: duplicates });
          setMessage("");
          setStatus("idle");
          return;
        }
      } catch (error) {
        setStatus("error");
        setMessage(error instanceof Error ? error.message : "重复文件检查失败。");
        return;
      }
    }

    setPendingAction(null);
    setStatus("parsing");
    setMessage("解析中...");
    try {
      const result = await parseDocuments(targetFiles);
      setManualDocuments(
        result.documents.map((document) => ({
          source: document.source,
          chunks: createInitialChunks(document, chunkConfig),
        })),
      );
      setSelectedDocumentIndex(0);
      setStatus("idle");
      setMessage("解析完成。");
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "解析失败。");
    }
  }

  async function runManualIngest(
    overwriteExisting = false,
    targetDocuments = manualDocuments,
    targetConfig = chunkConfig,
  ) {
    const documents = targetDocuments.map((document) => ({
      source: document.source,
      chunks: document.chunks.map((chunk) => chunk.text).filter((text) => text.trim()),
    }));
    if (documents.every((document) => document.chunks.length === 0)) {
      setStatus("error");
      setMessage("没有可写入的 chunk。");
      return;
    }

    const names = targetDocuments.map((document) => document.source);
    const hasApprovedOverwrite = names.some((name) => approvedOverwriteSources.includes(name));
    if (!overwriteExisting && !hasApprovedOverwrite) {
      try {
        const duplicates = await getExistingDuplicates(names);
        if (duplicates.length > 0) {
          setPendingAction({
            kind: "manual",
            documents: targetDocuments,
            duplicateSources: duplicates,
            chunkConfig: targetConfig,
          });
          setMessage("");
          setStatus("idle");
          return;
        }
      } catch (error) {
        setStatus("error");
        setMessage(error instanceof Error ? error.message : "重复文件检查失败。");
        return;
      }
    }

    setPendingAction(null);
    setStatus("uploading");
    setMessage(overwriteExisting || hasApprovedOverwrite ? "覆盖写入中..." : "写入中...");
    try {
      const result = await ingestManualChunks(
        documents,
        targetConfig,
        overwriteExisting || hasApprovedOverwrite,
      );
      setApprovedOverwriteSources([]);
      setStatus("done");
      setMessage(`写入 ${result.inserted} 个分块。`);
      onIngested(result);
    } catch (error) {
      if (error instanceof DuplicateSourceError) {
        setPendingAction({
          kind: "manual",
          documents: targetDocuments,
          duplicateSources: error.duplicateSources,
          chunkConfig: targetConfig,
        });
        setStatus("idle");
        setMessage("");
        return;
      }
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "入库失败。");
    }
  }

  function confirmPendingAction() {
    if (!pendingAction) {
      return;
    }
    if (pendingAction.kind === "auto") {
      void runAutoUpload(true, pendingAction.files);
      return;
    }
    if (pendingAction.kind === "parse") {
      setApprovedOverwriteSources(pendingAction.duplicateSources);
      void runParse(true, pendingAction.files);
      return;
    }
    void runManualIngest(true, pendingAction.documents, pendingAction.chunkConfig);
  }

  function cancelPendingAction() {
    setPendingAction(null);
    setStatus("idle");
    setMessage("");
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

  return (
    <section className="ingestPanel">
      <div className="panelHead">
        <div>
          <span className="panelKicker">Ingest</span>
          <h2>入库任务</h2>
        </div>
        <div className="modeSwitch" aria-label="入库模式">
          <button className={mode === "auto" ? "active" : ""} disabled={busy} onClick={() => setMode("auto")} type="button">
            自动
          </button>
          <button className={mode === "manual" ? "active" : ""} disabled={busy} onClick={() => setMode("manual")} type="button">
            手动
          </button>
        </div>
      </div>

      <label className="dropTarget">
        <input
          type="file"
          multiple
          accept=".md,.markdown,.txt,.pdf,.png,.jpg,.jpeg,.docx,.pptx,.xlsx,.html,.htm"
          onChange={handleFilesSelected}
        />
        <span className="dropGlyph">+</span>
        <span>
          <strong>{files.length > 0 ? `${files.length} 个文件` : "选择文件"}</strong>
          <small>{files.length > 0 ? `${(totalSize / 1024).toFixed(1)} KB` : "PDF / Office / HTML / Text"}</small>
        </span>
      </label>

      {files.length > 0 && (
        <div className="uploadQueue">
          {files.map((file, index) => (
            <div className="queueItem" key={`${file.name}-${file.size}-${index}`}>
              <div>
                <strong>{file.name}</strong>
                <span>
                  {(file.size / 1024).toFixed(1)} KB
                  {!SUPPORTED_EXTENSIONS.has(getFileExtension(file.name)) ? " / 不支持" : ""}
                </span>
              </div>
              <button disabled={busy} onClick={() => removeFile(index)} type="button">
                移除
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="chunkControls">
        <label>
          <span>Size</span>
          <input
            min={100}
            step={50}
            type="number"
            value={chunkConfig.chunk_size}
            onChange={(event) => updateChunkConfig("chunk_size", Number(event.target.value))}
          />
        </label>
        <label>
          <span>Overlap</span>
          <input
            min={0}
            step={20}
            type="number"
            value={chunkConfig.chunk_overlap}
            onChange={(event) => updateChunkConfig("chunk_overlap", Number(event.target.value))}
          />
        </label>
        <label>
          <span>Min</span>
          <input
            min={0}
            step={20}
            type="number"
            value={chunkConfig.min_chunk_size}
            onChange={(event) => updateChunkConfig("min_chunk_size", Number(event.target.value))}
          />
        </label>
      </div>

      {pendingAction && (
        <div className="confirmPanel" role="alert">
          <strong>文件名重复</strong>
          <span>{pendingAction.duplicateSources.join(" / ")}</span>
          <div>
            <button className="ghostButton" disabled={busy} onClick={cancelPendingAction} type="button">
              取消
            </button>
            <button disabled={busy} onClick={confirmPendingAction} type="button">
              覆盖
            </button>
          </div>
        </div>
      )}

      {mode === "auto" ? (
        <div className="primaryActions">
          <button disabled={!canStart || !chunkConfigValid} onClick={() => void runAutoUpload()} type="button">
            {status === "uploading" ? "写入中" : "解析并入库"}
          </button>
        </div>
      ) : (
        <div className="manualFlow">
          <div className="primaryActions two">
            <button disabled={!canStart || !chunkConfigValid} onClick={() => void runParse()} type="button">
              {status === "parsing" ? "解析中" : "解析"}
            </button>
            <button disabled={busy || manualDocuments.length === 0} onClick={() => void runManualIngest()} type="button">
              {status === "uploading" ? "写入中" : "入库"}
            </button>
          </div>
          {manualDocuments.length > 0 && (
            <div className="chunkWorkbench">
              <div className="chunkTabs">
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
              <div className="chunkToolbar">
                <strong>{manualChunkCount} chunks</strong>
                <button className="ghostButton" disabled={busy || !chunkConfigValid} onClick={regenerateManualChunks} type="button">
                  重分
                </button>
              </div>
              {selectedDocument && (
                <div className="chunkStack">
                  {selectedDocument.chunks.map((chunk, index) => (
                    <article className="chunkCard" key={chunk.id}>
                      <header>
                        <strong>#{index + 1}</strong>
                        <span>{chunk.text.trim().length}</span>
                      </header>
                      <textarea
                        disabled={busy}
                        rows={5}
                        value={chunk.text}
                        onChange={(event) =>
                          updateChunk(selectedDocumentIndex, chunk.id, event.target.value)
                        }
                      />
                      <footer>
                        <button className="ghostButton" disabled={busy} onClick={() => addChunk(selectedDocumentIndex, index)} type="button">
                          插入
                        </button>
                        <button className="ghostButton danger" disabled={busy || selectedDocument.chunks.length === 1} onClick={() => removeChunk(selectedDocumentIndex, chunk.id)} type="button">
                          删除
                        </button>
                      </footer>
                    </article>
                  ))}
                  <button className="ghostButton" disabled={busy} onClick={() => addChunk(selectedDocumentIndex)} type="button">
                    添加 chunk
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {unsupportedFiles.length > 0 && <p className="notice error">存在不支持的文件。</p>}
      {!chunkConfigValid && <p className="notice error">Chunk 参数无效。</p>}
      {message && <p className={`notice ${status}`}>{message}</p>}
    </section>
  );
}
