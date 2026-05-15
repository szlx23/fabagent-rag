import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { askQuestion, listDocuments } from "../api/rag";
import type { AskResponse, IngestedDocument } from "../types/rag";

const DEFAULT_QUESTION = "OPC 在半导体工艺中有哪些类型？";
const QUICK_PROMPTS = [
  "洁净室人员进入前需要遵守哪些要求？",
  "ICP-RIE 刻蚀报警时应该如何排查？",
  "MES Hold Lot 的处理流程是什么？",
  "SPC 报表中 Cpk、均值和标准差分别怎么看？",
];

const INTENT_LABELS: Record<AskResponse["intent"], string> = {
  lookup: "资料查询",
  summarize: "资料总结",
  chat: "闲聊",
};

function MarkdownBlock({ content }: { content: string }) {
  return (
    <div className="markdownBody answer">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}

function PlainTextBlock({ content }: { content: string }) {
  return <pre className="plainContext">{content}</pre>;
}

function normalizeText(text: unknown) {
  return String(text ?? "").replace(/\s+/g, " ").trim();
}

function getContextPreview(text: unknown) {
  const normalized = normalizeText(text);
  if (!normalized) {
    return "无文本内容";
  }
  return normalized.length > 96 ? `${normalized.slice(0, 96)}...` : normalized;
}

function getSourceLocation(context: AskResponse["contexts"][number]) {
  const parts = [String(context.source ?? "未知来源")];
  if (typeof context.page === "number") {
    parts.push(`第 ${context.page} 页`);
  }
  const sectionTitle = normalizeText(context.section_title);
  if (sectionTitle) {
    parts.push(sectionTitle);
  }
  return parts.join(" / ");
}

function getFileName(source: string, fileName?: string) {
  const trimmed = normalizeText(fileName);
  if (trimmed) {
    return trimmed;
  }
  const fallback = source.split(/[\\/]/).filter(Boolean).pop();
  return fallback || source;
}

function getFolderName(source: string, fileName: string) {
  const normalized = source.replace(/\\/g, "/");
  const slashIndex = normalized.lastIndexOf("/");
  if (slashIndex < 0) {
    return "根目录";
  }

  const folder = normalized.slice(0, slashIndex).trim();
  return folder && folder !== fileName ? folder : "根目录";
}

function matchesDocument(document: IngestedDocument, query: string) {
  if (!query) {
    return true;
  }

  const searchableText = [
    document.file_name ?? "",
    document.source,
    document.file_ext ?? "",
    document.parser ?? "",
    String(document.chunk_count ?? ""),
  ]
    .join(" ")
    .toLowerCase();

  return searchableText.includes(query.toLowerCase());
}

type AskPanelProps = {
  refreshKey?: string;
};

export function AskPanel({ refreshKey = "" }: AskPanelProps) {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(3);
  const [loading, setLoading] = useState(false);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [error, setError] = useState("");
  const [documentError, setDocumentError] = useState("");
  const [documents, setDocuments] = useState<IngestedDocument[]>([]);
  const [documentSearch, setDocumentSearch] = useState("");
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
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

  async function handleAsk() {
    const finalQuestion = question.trim() || DEFAULT_QUESTION;
    if (!finalQuestion) {
      setError("请输入问题。");
      return;
    }

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

  const selectedContext = result?.contexts[selectedContextIndex];
  const selectedSourceSet = useMemo(() => new Set(selectedSources), [selectedSources]);
  const selectedLabel =
    selectedSources.length > 0 ? `限定 ${selectedSources.length} 个文件` : "全库检索";

  const filteredDocuments = useMemo(
    () => documents.filter((document) => matchesDocument(document, documentSearch.trim())),
    [documentSearch, documents],
  );

  const selectedDocumentLabels = useMemo(
    () =>
      selectedSources
        .map((source) => documents.find((document) => document.source === source))
        .filter((document): document is IngestedDocument => Boolean(document))
        .map((document) => getFileName(document.source, document.file_name)),
    [documents, selectedSources],
  );

  function toggleSource(source: string) {
    setSelectedSources((current) =>
      current.includes(source)
        ? current.filter((item) => item !== source)
        : [...current, source],
    );
  }

  const documentCount = documents.length;
  const visibleCount = filteredDocuments.length;
  const hasSearch = documentSearch.trim().length > 0;

  return (
    <section className="panel answerPanel">
      <div className="panelHeader panelHeaderStack">
        <div className="panelTitleGroup">
          <span className="panelKicker">Knowledge scope</span>
          <h2>检索问答</h2>
          <p className="panelLead">按文件名搜索、限定资料范围，再发起检索。</p>
        </div>
        <div className="panelMetrics">
          <div className="metricCard">
            <span>文档</span>
            <strong>{documentCount}</strong>
          </div>
          <div className="metricCard">
            <span>当前可见</span>
            <strong>{visibleCount}</strong>
          </div>
          <div className="metricCard">
            <span>范围</span>
            <strong>{selectedLabel}</strong>
          </div>
        </div>
      </div>

      <div className="askComposer">
        <div className="documentScopePanel">
          <div className="documentScopeHeader">
            <div className="documentScopeHeaderText">
              <strong>查询范围</strong>
              <span>
                {documentCount > 0
                  ? hasSearch
                    ? `搜索到 ${visibleCount} 个文件`
                    : `${documentCount} 个已入库文件`
                  : "暂无已入库文件"}
              </span>
            </div>
            <div className="documentScopeActions">
              {selectedSources.length > 0 && (
                <button
                  className="secondaryButton"
                  disabled={loading}
                  onClick={() => setSelectedSources([])}
                  type="button"
                >
                  清空范围
                </button>
              )}
              <button
                className="secondaryButton"
                disabled={documentsLoading || loading}
                onClick={loadDocuments}
                type="button"
              >
                {documentsLoading ? "刷新中" : "刷新"}
              </button>
            </div>
          </div>

          <div className="documentScopeToolbar">
            <label className="documentSearchBox">
              <span>搜索</span>
              <input
                aria-label="搜索已入库文件"
                placeholder="按文件名、路径、后缀或解析器搜索"
                value={documentSearch}
                onChange={(event) => setDocumentSearch(event.target.value)}
              />
            </label>
            <div className="documentScopeChipRow" aria-label="已选文件">
              {selectedDocumentLabels.length > 0 ? (
                selectedDocumentLabels.map((label) => (
                  <span className="scopeChip" key={label}>
                    {label}
                  </span>
                ))
              ) : (
                <span className="scopeChip muted">未限定文件</span>
              )}
            </div>
          </div>

          {documentError && <p className="scopeMessage error">{documentError}</p>}
          {!documentError && documentCount === 0 && (
            <p className="scopeMessage">暂无已入库文件</p>
          )}
          {!documentError && documentCount > 0 && filteredDocuments.length === 0 && (
            <p className="scopeMessage">没有匹配的文件名，换个关键词试试。</p>
          )}
          {!documentError && filteredDocuments.length > 0 && (
            <div className="documentScopeList" aria-label="已入库文件">
              {filteredDocuments.map((document) => {
                const active = selectedSourceSet.has(document.source);
                const fileName = getFileName(document.source, document.file_name);
                const folderName = getFolderName(document.source, fileName);

                return (
                  <button
                    className={`documentScopeItem ${active ? "active" : ""}`}
                    disabled={loading}
                    key={document.source}
                    onClick={() => toggleSource(document.source)}
                    type="button"
                    title={document.source}
                  >
                    <span className="documentScopeName">{fileName}</span>
                    <span className="documentScopePath">{folderName}</span>
                    <span className="documentScopeMeta">
                      {document.chunk_count} 块
                      {document.file_ext ? ` · ${document.file_ext}` : ""}
                      {document.parser ? ` · ${document.parser}` : ""}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="questionBox">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder={DEFAULT_QUESTION}
            rows={4}
          />
        </div>

        <div className="quickPromptRow" aria-label="常用问题">
          {QUICK_PROMPTS.map((prompt) => (
            <button
              className="promptButton"
              disabled={loading}
              key={prompt}
              onClick={() => setQuestion(prompt)}
              type="button"
            >
              {prompt}
            </button>
          ))}
        </div>

        <div className="askToolbar">
          <label>
            <span>召回</span>
            <input
              min={1}
              max={20}
              type="number"
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            />
          </label>
          <button disabled={loading} onClick={handleAsk} type="button">
            {loading ? "查询中" : "提问"}
          </button>
        </div>
      </div>

      {error && <p className="statusText error">{error}</p>}

      {!result && !error && (
        <div className="answerEmpty">
          <strong>等待提问</strong>
          <span>可直接使用默认问题，或先限定文件范围。</span>
        </div>
      )}

      {result && (
        <div className="resultBlock">
          <div className="sectionTitleRow">
            <h3>回答</h3>
            <span>
              {INTENT_LABELS[result.intent] ?? result.intent} · {result.contexts.length} 条召回
            </span>
          </div>
          <MarkdownBlock content={result.answer} />

          <div className="sectionTitleRow">
            <h3>引用来源</h3>
            <span>点击查看原文</span>
          </div>
          {result.contexts.length > 0 ? (
            <div className="contextExplorer">
              <div className="contextSources" role="list">
                {result.contexts.map((context, index) => {
                  const active = index === selectedContextIndex;
                  return (
                    <button
                      className={`contextSourceButton ${active ? "active" : ""}`}
                      key={`${String(context.source ?? "unknown")}-${index}`}
                      onClick={() => setSelectedContextIndex(index)}
                      type="button"
                    >
                      <span className="contextSourceTop">
                        <strong>{String(context.source ?? "未知来源")}</strong>
                        {typeof context.score === "number" && (
                          <small>{context.score.toFixed(4)}</small>
                        )}
                      </span>
                      <span className="contextSourceIndex">{getSourceLocation(context)}</span>
                      <span className="contextPreview">{getContextPreview(context.text)}</span>
                    </button>
                  );
                })}
              </div>

              {selectedContext && (
                <article className="contextDetail">
                  <div className="contextDetailHeader">
                    <div>
                      <strong>{String(selectedContext.source ?? "未知来源")}</strong>
                      {typeof selectedContext.page === "number" && (
                        <span>第 {selectedContext.page} 页</span>
                      )}
                      {String(selectedContext.section_title ?? "").trim() && (
                        <span>{String(selectedContext.section_title)}</span>
                      )}
                    </div>
                    {typeof selectedContext.score === "number" && (
                      <small>分数 {selectedContext.score.toFixed(4)}</small>
                    )}
                  </div>
                  <PlainTextBlock content={String(selectedContext.text ?? "")} />
                </article>
              )}
            </div>
          ) : (
            <p className="emptyState">没有召回上下文。</p>
          )}
        </div>
      )}
    </section>
  );
}
