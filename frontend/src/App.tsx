import { useMemo, useState } from "react";

import { AskPanel } from "./components/AskPanel";
import { FileUploadPanel } from "./components/FileUploadPanel";
import type { IngestResponse } from "./types/rag";

function App() {
  const [lastIngest, setLastIngest] = useState<IngestResponse | null>(null);
  const [refreshKey, setRefreshKey] = useState("init");

  function handleIngested(result: IngestResponse) {
    setLastIngest(result);
    setRefreshKey(`ingest-${Date.now()}`);
  }

  function handleDocumentsChanged() {
    setRefreshKey(`docs-${Date.now()}`);
  }

  const statusCards = useMemo(
    () => [
      {
        label: "最近写入",
        value: lastIngest ? String(lastIngest.inserted) : "0",
      },
      {
        label: "来源文件",
        value: lastIngest ? String(lastIngest.sources.length) : "0",
      },
      {
        label: "文档数",
        value: lastIngest ? String(lastIngest.documents) : "0",
      },
    ],
    [lastIngest],
  );

  return (
    <main className="productShell">
      <header className="topBar">
        <div>
          <span className="brandMark">FA</span>
          <div>
            <strong>FabAgent RAG</strong>
            <small>Knowledge Operations</small>
          </div>
        </div>
        <dl className="topStats" aria-label="运行状态">
          {statusCards.map((card) => (
            <div key={card.label}>
              <dt>{card.label}</dt>
              <dd>{card.value}</dd>
            </div>
          ))}
        </dl>
      </header>

      <section className="workSurface">
        <FileUploadPanel onIngested={handleIngested} />
        <AskPanel refreshKey={refreshKey} onDocumentsChanged={handleDocumentsChanged} />
      </section>
    </main>
  );
}

export default App;
