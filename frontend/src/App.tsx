import { useState } from "react";

import { AskPanel } from "./components/AskPanel";
import { FileUploadPanel } from "./components/FileUploadPanel";
import type { IngestResponse } from "./types/rag";

function App() {
  const [lastIngest, setLastIngest] = useState<IngestResponse | null>(null);

  return (
    <main className="appShell">
      <header className="topBar">
        <div className="brandBlock">
          <span className="eyebrow">FabAgent Workspace</span>
          <h1>RAG 文档工作台</h1>
          <p>上传资料、建立索引、围绕来源追问。</p>
        </div>
        <div className="topStats" aria-label="服务模块">
          <div>
            <span>API</span>
            <strong>FastAPI</strong>
          </div>
          <div>
            <span>Vector DB</span>
            <strong>Milvus</strong>
          </div>
          <div>
            <span>Parser</span>
            <strong>MinerU+</strong>
          </div>
        </div>
      </header>

      {lastIngest && (
        <section className="summaryStrip">
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
