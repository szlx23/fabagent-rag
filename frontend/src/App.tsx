import { useState } from "react";

import { AskPanel } from "./components/AskPanel";
import { FileUploadPanel } from "./components/FileUploadPanel";
import type { IngestResponse } from "./types/rag";

function App() {
  const [lastIngest, setLastIngest] = useState<IngestResponse | null>(null);

  return (
    <main className="appShell">
      <header className="heroBar">
        <div className="brandBlock">
          <span className="eyebrow">FabAgent RAG Console</span>
          <h1>文档索引与问答工作台</h1>
          <p>把工艺、设备、SOP 和表格资料变成可追溯的知识索引。</p>
        </div>
        <aside className="systemPanel" aria-label="服务模块">
          <div className="systemStatus">
            <span className="statusDot" aria-hidden="true" />
            Local stack
          </div>
          <dl className="systemGrid">
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

      <section className="metricsGrid" aria-label="工作流状态">
        <article className="metricCard">
          <span>Document Flow</span>
          <strong>Upload / Parse / Index</strong>
          <small>支持自动入库与手动 chunk 审核。</small>
        </article>
        <article className="metricCard">
          <span>Answer Grounding</span>
          <strong>Source-first Retrieval</strong>
          <small>回答与召回来源分开展示。</small>
        </article>
        <article className="metricCard accent">
          <span>Last Ingest</span>
          <strong>{lastIngest ? `${lastIngest.inserted} chunks` : "Waiting"}</strong>
          <small>
            {lastIngest
              ? `${lastIngest.documents} 个文档：${lastIngest.sources.join("，")}`
              : "完成入库后会在这里显示最近写入结果。"}
          </small>
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
        <AskPanel />
      </div>
    </main>
  );
}

export default App;
