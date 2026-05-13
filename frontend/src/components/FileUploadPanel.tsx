import type { ChangeEvent } from "react";
import { useMemo, useState } from "react";

import { uploadDocuments } from "../api/rag";
import type { IngestResponse } from "../types/rag";

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

type FileUploadPanelProps = {
  onIngested: (result: IngestResponse) => void;
};

function getFileExtension(fileName: string) {
  const lastDot = fileName.lastIndexOf(".");
  return lastDot >= 0 ? fileName.slice(lastDot).toLowerCase() : "";
}

export function FileUploadPanel({ onIngested }: FileUploadPanelProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [batchSize, setBatchSize] = useState(10);
  const [status, setStatus] = useState<"idle" | "uploading" | "done" | "error">("idle");
  const [message, setMessage] = useState("");

  const unsupportedFiles = useMemo(
    () => files.filter((file) => !SUPPORTED_EXTENSIONS.has(getFileExtension(file.name))),
    [files],
  );
  const totalSize = useMemo(
    () => files.reduce((total, file) => total + file.size, 0),
    [files],
  );
  const canUpload = files.length > 0 && unsupportedFiles.length === 0 && status !== "uploading";

  function handleFilesSelected(event: ChangeEvent<HTMLInputElement>) {
    const selectedFiles = Array.from(event.target.files ?? []);
    setFiles((currentFiles) => [...currentFiles, ...selectedFiles]);
    setStatus("idle");
    setMessage("");

    // 清空 input 的值，用户删除后可以重新选择同一个文件。
    event.target.value = "";
  }

  function removeFile(fileIndex: number) {
    setFiles((currentFiles) => currentFiles.filter((_, index) => index !== fileIndex));
    setStatus("idle");
    setMessage("");
  }

  async function handleUpload() {
    if (files.length === 0) {
      setStatus("error");
      setMessage("请选择至少一个文件。");
      return;
    }

    if (unsupportedFiles.length > 0) {
      setStatus("error");
      setMessage(`请先删除不支持的文件：${unsupportedFiles.map((file) => file.name).join("，")}`);
      return;
    }

    setStatus("uploading");
    setMessage("正在解析并写入向量库...");

    try {
      const result = await uploadDocuments(files, batchSize);
      setStatus("done");
      setMessage(`已写入 ${result.inserted} 个分块，来源 ${result.documents} 个文件。`);
      onIngested(result);
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "上传失败。");
    }
  }

  return (
    <section className="panel">
      <div className="panelHeader">
        <div>
          <h2>文档入库</h2>
          <p>上传 Markdown、PDF、图片或 Office 文档，后端会解析后写入 Milvus。</p>
        </div>
      </div>

      <label className="dropZone">
        <input
          type="file"
          multiple
          accept=".md,.markdown,.txt,.pdf,.png,.jpg,.jpeg,.docx,.pptx,.xlsx,.html,.htm"
          onChange={handleFilesSelected}
        />
        <span>选择文件</span>
        <strong>{files.length > 0 ? `${files.length} 个文件已选择` : "支持多文件上传"}</strong>
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
                  disabled={status === "uploading"}
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

      <div className="controlRow">
        <label>
          批大小
          <input
            min={1}
            max={100}
            type="number"
            value={batchSize}
            onChange={(event) => setBatchSize(Number(event.target.value))}
          />
        </label>
        <button disabled={!canUpload} onClick={handleUpload} type="button">
          {status === "uploading" ? "处理中" : "上传入库"}
        </button>
      </div>

      {unsupportedFiles.length > 0 && (
        <p className="statusText error">存在不支持的文件，请删除后再上传。</p>
      )}
      {message && <p className={`statusText ${status}`}>{message}</p>}
    </section>
  );
}
