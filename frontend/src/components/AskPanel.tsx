import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { askQuestion, listDocuments } from "../api/rag";
import type { AskResponse, IngestedDocument } from "../types/rag";

const DEFAULT_QUESTION = "这些文档主要包含哪些半导体制造或设备操作信息？";
const QUICK_PROMPTS = [
  "总结当前资料的核心内容",
  "有哪些关键操作步骤？",
  "列出涉及的参数和限制",
];

const INTENT_LABELS: Record<AskResponse["intent"], string> = {
  lookup: "资料查询",
  summarize: "资料总结",
  chat: "闲聊",
};

type MarkdownBlockProps = {
  content: string;
  variant?: "answer" | "context";
};

function MarkdownBlock({ content, variant = "answer" }: MarkdownBlockProps) {
  return (
    <div className={`markdownBody ${variant}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}

function getContextPreview(text: unknown) {
  const normalized = String(text ?? "").replace(/\s+/g, " ").trim();
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
  const sectionTitle = String(context.section_title ?? "").trim();
  if (sectionTitle) {
    parts.push(sectionTitle);
  }
  return parts.join(" / ");
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

  function toggleSource(source: string) {
    setSelectedSources((current) =>
      current.includes(source)
        ? current.filter((item) => item !== source)
        : [...current, source],
    );
  }

  return (
    <section className="panel answerPanel">
      <div className="panelHeader">
        <div>
          <span className="panelLabel">Retrieval console</span>
          <h2>检索问答</h2>
          <p>问题、答案和召回来源在同一视图里核对。</p>
        </div>
        <span className="panelBadge">Grounded QA</span>
      </div>

      <div className="askComposer">
        <div className="documentScopePanel">
          <div className="documentScopeHeader">
            <div>
              <strong>查询范围</strong>
              <span>{selectedLabel}</span>
            </div>
            <div className="documentScopeActions">
              {selectedSources.length > 0 && (
                <button
                  className="secondaryButton"
                  disabled={loading}
                  onClick={() => setSelectedSources([])}
                  type="button"
                >
                  清空选择
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

          {documentError && <p className="scopeMessage error">{documentError}</p>}
          {!documentError && documents.length === 0 && (
            <p className="scopeMessage">暂无已入库文件，完成入库后刷新列表。</p>
          )}
          {documents.length > 0 && (
            <div className="documentScopeList" aria-label="已入库文件">
              {documents.map((document) => {
                const active = selectedSourceSet.has(document.source);
                return (
                  <button
                    className={`documentScopeItem ${active ? "active" : ""}`}
                    disabled={loading}
                    key={document.source}
                    onClick={() => toggleSource(document.source)}
                    type="button"
                  >
                    <span className="documentScopeName">{document.source}</span>
                    <span className="documentScopeMeta">
                      {document.chunk_count} chunks
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
            <span>Top K</span>
            <input
              min={1}
              max={20}
              type="number"
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
            />
          </label>
          <button disabled={loading} onClick={handleAsk} type="button">
            {loading ? "查询中" : "发送问题"}
          </button>
        </div>
      </div>

      {error && <p className="statusText error">{error}</p>}

      {!result && !error && (
        <div className="answerEmpty">
          <strong>等待提问</strong>
          <span>入库后可以直接使用默认问题，也可以从上方快捷问题开始。</span>
        </div>
      )}

      {result && (
        <div className="resultBlock">
          <div className="sectionTitleRow">
            <h3>回答</h3>
            <span>
              {INTENT_LABELS[result.intent] ?? result.intent} · {result.contexts.length} 条召回 ·{" "}
              {selectedLabel}
            </span>
          </div>
          <MarkdownBlock content={result.answer} variant="answer" />

          <div className="sectionTitleRow">
            <h3>引用来源</h3>
            <span>点击来源查看原文片段</span>
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
                      <small>score {selectedContext.score.toFixed(4)}</small>
                    )}
                  </div>
                  <MarkdownBlock content={String(selectedContext.text ?? "")} variant="context" />
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
