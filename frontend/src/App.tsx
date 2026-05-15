import { useState } from "react";

import { AskPanel } from "./components/AskPanel";
import { FileUploadPanel } from "./components/FileUploadPanel";
import type { IngestResponse } from "./types/rag";

function App() {
  const [lastIngest, setLastIngest] = useState<IngestResponse | null>(null);

  return (
    <main className="appShell">
      <header className="masthead">
        <div className="brandBlock">
          <span className="eyebrow">FabAgent RAG</span>
          <h1>文档知识工作台</h1>
          <p>上传资料、审核分块、检索问答，并始终保留答案来源。</p>
        </div>
        <aside className="runtimePanel" aria-label="运行环境">
          <div className="runtimeStatus">
            <span className="statusDot" aria-hidden="true" />
            Local runtime
          </div>
          <dl className="runtimeGrid">
            <div>
              <dt>API</dt>
              <dd>FastAPI</dd>
            </div>
            <div>
              <dt>Vector DB</dt>
              <dd>Milvus</dd>
            </div>
            <div>
              <dt>Parser</dt>
              <dd>MinerU+</dd>
            </div>
          </dl>
        </aside>
      </header>

      <section className="workflowStrip" aria-label="工作流状态">
        <article>
          <span>01</span>
          <div>
            <strong>解析文件</strong>
            <small>PDF、Office、表格、Markdown、HTML、图片</small>
          </div>
        </article>
        <article>
          <span>02</span>
          <div>
            <strong>确认分块</strong>
            <small>自动入库或人工检查 chunk 边界</small>
          </div>
        </article>
        <article className="workflowStripResult">
          <span>03</span>
          <div>
            <strong>{lastIngest ? `${lastIngest.inserted} 个 chunk 已入库` : "等待入库"}</strong>
            <small>
              {lastIngest
                ? `${lastIngest.documents} 个文档：${lastIngest.sources.join("，")}`
                : "完成后可以直接在右侧提问。"}
            </small>
          </div>
        </article>
      </section>

      {lastIngest && (
        <section className="summaryStrip" aria-label="最近入库结果">
          <div>
            <span>最近入库</span>
            <strong>{lastIngest.documents} 个文档</strong>
          </div>
          <div>
            <span>写入分块</span>
            <strong>{lastIngest.inserted}</strong>
          </div>
          <div className="summarySources">
            <span>来源</span>
            <strong>{lastIngest.sources.join("，")}</strong>
          </div>
        </section>
      )}

      <div className="workspaceGrid">
        <FileUploadPanel onIngested={setLastIngest} />
        <AskPanel
          refreshKey={lastIngest ? `${lastIngest.inserted}-${lastIngest.sources.join("|")}` : ""}
        />
      </div>
    </main>
  );
}

export default App;
