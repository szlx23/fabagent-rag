import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { askQuestion, deleteDocuments, listDocuments } from "../api/rag";
import type { AskResponse, IngestedDocument } from "../types/rag";

const DEFAULT_QUESTION = "OPC 在半导体工艺中有哪些类型？";
const QUICK_PROMPTS = [
  "洁净室人员进入前需要遵守哪些要求？",
  "ICP-RIE 刻蚀报警时应该如何排查？",
  "MES Hold Lot 的处理流程是什么？",
  "SPC 报表中 Cpk、均值和标准差分别怎么看？",
];

const INTENT_LABELS: Record<AskResponse["intent"], string> = {
  lookup: "Lookup",
  summarize: "Summary",
  chat: "Chat",
};

type AskPanelProps = {
  refreshKey?: string;
  onDocumentsChanged?: () => void;
};

function normalizeText(text: unknown) {
  return String(text ?? "").replace(/\s+/g, " ").trim();
}

function getFileName(source: string, fileName?: string) {
  const trimmed = normalizeText(fileName);
  if (trimmed) {
    return trimmed;
  }
  return source.split(/[\\/]/).filter(Boolean).pop() || source;
}

function matchesDocument(document: IngestedDocument, query: string) {
  if (!query) {
    return true;
  }
  return [
    document.file_name ?? "",
    document.source,
    document.file_ext ?? "",
    document.parser ?? "",
    String(document.chunk_count ?? ""),
  ]
    .join(" ")
    .toLowerCase()
    .includes(query.toLowerCase());
}

function getContextPreview(text: unknown) {
  const normalized = normalizeText(text);
  if (!normalized) {
    return "No text";
  }
  return normalized.length > 120 ? `${normalized.slice(0, 120)}...` : normalized;
}

function MarkdownBlock({ content }: { content: string }) {
  return (
    <div className="markdownBody">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}

export function AskPanel({ refreshKey = "", onDocumentsChanged }: AskPanelProps) {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(4);
  const [loading, setLoading] = useState(false);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [deletingSource, setDeletingSource] = useState("");
  const [error, setError] = useState("");
  const [documentError, setDocumentError] = useState("");
  const [documents, setDocuments] = useState<IngestedDocument[]>([]);
  const [documentSearch, setDocumentSearch] = useState("");
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [pendingDelete, setPendingDelete] = useState<IngestedDocument | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [selectedContextIndex, setSelectedContextIndex] = useState(0);

  async function loadDocuments() {
    setDocumentsLoading(true);
    setDocumentError("");
    try {
      const response = await listDocuments();
      setDocuments(response.documents);
      setSelectedSources((current) =>
        current.filter((source) =>
          response.documents.some((document) => document.source === source),
        ),
      );
    } catch (caught) {
      setDocumentError(caught instanceof Error ? caught.message : "文档列表加载失败。");
    } finally {
      setDocumentsLoading(false);
    }
  }

  useEffect(() => {
    void loadDocuments();
  }, [refreshKey]);

  async function handleDelete(document: IngestedDocument) {
    setDeletingSource(document.source);
    setDocumentError("");
    try {
      await deleteDocuments([document.source]);
      setPendingDelete(null);
      setSelectedSources((current) => current.filter((source) => source !== document.source));
      await loadDocuments();
      onDocumentsChanged?.();
    } catch (caught) {
      setDocumentError(caught instanceof Error ? caught.message : "删除失败。");
    } finally {
      setDeletingSource("");
    }
  }

  async function handleAsk() {
    const finalQuestion = question.trim() || DEFAULT_QUESTION;
    setLoading(true);
    setError("");
    try {
      const response = await askQuestion(finalQuestion, topK, selectedSources);
      setResult(response);
      setSelectedContextIndex(0);
    } catch (caught) {
      setResult(null);
      setError(caught instanceof Error ? caught.message : "查询失败。");
    } finally {
      setLoading(false);
    }
  }

  function toggleSource(source: string) {
    setSelectedSources((current) =>
      current.includes(source)
        ? current.filter((item) => item !== source)
        : [...current, source],
    );
  }

  const filteredDocuments = useMemo(
    () => documents.filter((document) => matchesDocument(document, documentSearch.trim())),
    [documentSearch, documents],
  );
  const selectedSourceSet = useMemo(() => new Set(selectedSources), [selectedSources]);
  const selectedDocumentNames = useMemo(
    () =>
      selectedSources
        .map((source) => documents.find((document) => document.source === source))
        .filter((document): document is IngestedDocument => Boolean(document))
        .map((document) => getFileName(document.source, document.file_name)),
    [documents, selectedSources],
  );
  const scopeLabel =
    selectedDocumentNames.length > 0
      ? `将基于 ${selectedDocumentNames.slice(0, 3).join("、")}${
          selectedDocumentNames.length > 3 ? ` 等 ${selectedDocumentNames.length} 个文件` : ""
        } 回复`
      : "将基于全部已入库文件回复";
  const selectedContext = result?.contexts[selectedContextIndex];

  return (
    <section className="queryPanel">
      <div className="libraryPane">
        <div className="panelHead">
          <div>
            <span className="panelKicker">Library</span>
            <h2>知识库</h2>
          </div>
          <button className="ghostButton" disabled={documentsLoading} onClick={loadDocuments} type="button">
            {documentsLoading ? "刷新中" : "刷新"}
          </button>
        </div>

        <div className="libraryToolbar">
          <input
            aria-label="搜索文件"
            placeholder="搜索文件"
            value={documentSearch}
            onChange={(event) => setDocumentSearch(event.target.value)}
          />
          {selectedSources.length > 0 && (
            <button className="ghostButton" disabled={loading} onClick={() => setSelectedSources([])} type="button">
              清空
            </button>
          )}
        </div>

        {documentError && <p className="notice error">{documentError}</p>}
        {!documentError && filteredDocuments.length === 0 && (
          <p className="notice idle">{documents.length === 0 ? "暂无文件。" : "没有匹配文件。"}</p>
        )}
        <div className="libraryList" aria-label="已入库文件">
          {filteredDocuments.map((document) => {
            const active = selectedSourceSet.has(document.source);
            const fileName = getFileName(document.source, document.file_name);
            return (
              <article className={`libraryItem ${active ? "active" : ""}`} key={document.source}>
                <button disabled={loading} onClick={() => toggleSource(document.source)} type="button">
                  <strong>{fileName}</strong>
                  <span>
                    {document.chunk_count} chunks
                    {document.file_ext ? ` / ${document.file_ext}` : ""}
                  </span>
                </button>
                <div className="libraryItemActions">
                  <button
                    className="iconButton danger"
                    disabled={deletingSource === document.source}
                    onClick={() => setPendingDelete(document)}
                    type="button"
                    title="删除"
                  >
                    删除
                  </button>
                </div>
              </article>
            );
          })}
        </div>

        {pendingDelete && (
          <div className="confirmPanel compact" role="alert">
            <strong>{getFileName(pendingDelete.source, pendingDelete.file_name)}</strong>
            <span>删除后将从检索范围移除。</span>
            <div>
              <button className="ghostButton" disabled={Boolean(deletingSource)} onClick={() => setPendingDelete(null)} type="button">
                取消
              </button>
              <button className="dangerButton" disabled={Boolean(deletingSource)} onClick={() => void handleDelete(pendingDelete)} type="button">
                删除
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="askPane">
        <div className="panelHead">
          <div>
            <span className="panelKicker">Ask</span>
            <h2>检索问答</h2>
          </div>
          <span className="scopePill">
            {selectedSources.length > 0 ? `${selectedSources.length} files` : "All files"}
          </span>
        </div>

        <textarea
          className="questionInput"
          rows={5}
          value={question}
          placeholder={DEFAULT_QUESTION}
          onChange={(event) => setQuestion(event.target.value)}
        />

        <div className="scopeNotice" title={scopeLabel}>
          <span>{scopeLabel}</span>
        </div>

        <div className="promptStrip">
          {QUICK_PROMPTS.map((prompt) => (
            <button className="ghostButton" disabled={loading} key={prompt} onClick={() => setQuestion(prompt)} type="button">
              {prompt}
            </button>
          ))}
        </div>

        <div className="askActions">
          <label>
            <span>Top K</span>
            <input min={1} max={20} type="number" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
          </label>
          <button disabled={loading} onClick={handleAsk} type="button">
            {loading ? "查询中" : "提问"}
          </button>
        </div>

        {error && <p className="notice error">{error}</p>}

        {result ? (
          <div className="answerWorkspace">
            <div className="answerHeader">
              <strong>{INTENT_LABELS[result.intent] ?? result.intent}</strong>
              <span>{result.contexts.length} contexts</span>
            </div>
            <MarkdownBlock content={result.answer} />

            {result.contexts.length > 0 && (
              <div className="contextGrid">
                <div className="contextList">
                  {result.contexts.map((context, index) => (
                    <button
                      className={index === selectedContextIndex ? "active" : ""}
                      key={`${String(context.source ?? "unknown")}-${index}`}
                      onClick={() => setSelectedContextIndex(index)}
                      type="button"
                    >
                      <strong>{getFileName(String(context.source ?? "未知来源"))}</strong>
                      <span>{getContextPreview(context.text)}</span>
                    </button>
                  ))}
                </div>
                {selectedContext && (
                  <article className="contextDetail">
                    <header>
                      <strong>{getFileName(String(selectedContext.source ?? "未知来源"))}</strong>
                      {typeof selectedContext.score === "number" && (
                        <span>{selectedContext.score.toFixed(4)}</span>
                      )}
                    </header>
                    <pre>{String(selectedContext.text ?? "")}</pre>
                  </article>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="emptyAnswer">
            <strong>Ready</strong>
            <span>{documents.length} files indexed</span>
          </div>
        )}
      </div>
    </section>
  );
}
