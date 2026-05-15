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
          <h1>文档问答工作台</h1>
        </div>
        <aside className="runtimePanel" aria-label="运行环境">
          <div className="runtimeStatus">
            <span className="statusDot" aria-hidden="true" />
            本地服务
          </div>
          <dl className="runtimeGrid">
            <div>
              <dt>接口</dt>
              <dd>FastAPI</dd>
            </div>
            <div>
              <dt>向量库</dt>
              <dd>Milvus</dd>
            </div>
            <div>
              <dt>解析</dt>
              <dd>多格式</dd>
            </div>
          </dl>
        </aside>
      </header>

      <section className="workflowStrip" aria-label="工作流状态">
        <article>
          <span>1</span>
          <div>
            <strong>上传</strong>
            <small>多格式资料</small>
          </div>
        </article>
        <article>
          <span>2</span>
          <div>
            <strong>入库</strong>
            <small>自动或手动分块</small>
          </div>
        </article>
        <article className="workflowStripResult">
          <span>3</span>
          <div>
            <strong>{lastIngest ? `${lastIngest.inserted} 个分块` : "待查询"}</strong>
            <small>{lastIngest ? `${lastIngest.documents} 个文档已入库` : "选择资料后开始"}</small>
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
