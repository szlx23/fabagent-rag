import { useState } from "react";

import { AskPanel } from "./components/AskPanel";
import { FileUploadPanel } from "./components/FileUploadPanel";
import type { IngestResponse } from "./types/rag";

function App() {
  const [lastIngest, setLastIngest] = useState<IngestResponse | null>(null);

  return (
    <main className="appShell">
      <header className="topBar">
        <div>
          <h1>fabagent RAG</h1>
          <p>文档上传、解析入库和检索问答。</p>
        </div>
        <div className="serviceBadge">FastAPI / Milvus / MinerU</div>
      </header>

      {lastIngest && (
        <section className="summaryStrip">
          <strong>最近入库</strong>
          <span>{lastIngest.documents} 个文档</span>
          <span>{lastIngest.inserted} 个分块</span>
          <span>{lastIngest.sources.join("，")}</span>
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
