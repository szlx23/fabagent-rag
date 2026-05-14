import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { askQuestion } from "../api/rag";
import type { AskResponse } from "../types/rag";

const DEFAULT_QUESTION = "这些文档主要包含哪些半导体制造或设备操作信息？";
const QUICK_PROMPTS = [
  "总结当前资料的核心内容",
  "有哪些关键操作步骤？",
  "列出涉及的参数和限制",
];

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

export function AskPanel() {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(3);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [selectedContextIndex, setSelectedContextIndex] = useState(0);

  async function handleAsk() {
    const finalQuestion = question.trim() || DEFAULT_QUESTION;
    if (!finalQuestion) {
      setError("请输入问题。");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await askQuestion(finalQuestion, topK);
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

  return (
    <section className="panel answerPanel">
      <div className="panelHeader">
        <div>
          <span className="panelLabel">Retrieval</span>
          <h2>检索问答</h2>
          <p>回答和引用来源分开展示，方便核对依据。</p>
        </div>
        <span className="panelBadge">Grounded QA</span>
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

      <div className="controlRow">
        <label>
          Top K
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

      {error && <p className="statusText error">{error}</p>}

      {result && (
        <div className="resultBlock">
          <div className="sectionTitleRow">
            <h3>回答</h3>
            <span>{result.contexts.length} 条召回</span>
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
