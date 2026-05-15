import { useMemo, useState } from "react";

import { AskPanel } from "./components/AskPanel";
import { FileUploadPanel } from "./components/FileUploadPanel";
import type { IngestResponse } from "./types/rag";

function App() {
  const [lastIngest, setLastIngest] = useState<IngestResponse | null>(null);

  const overviewCards = useMemo(
    () => [
      {
        label: "已写入分块",
        value: lastIngest ? String(lastIngest.inserted) : "0",
        hint: lastIngest ? `${lastIngest.documents} 个文档` : "等待首次入库",
      },
      {
        label: "入库来源",
        value: lastIngest ? String(lastIngest.sources.length) : "0",
        hint: lastIngest ? "最近批次已同步" : "尚未产生来源",
      },
      {
        label: "当前状态",
        value: lastIngest ? "Ready" : "Idle",
        hint: lastIngest ? "可以立即检索" : "上传后即可开始问答",
      },
    ],
    [lastIngest],
  );

  return (
    <main className="appShell">
      <div className="appBackdrop" aria-hidden="true">
        <span className="orb orbOne" />
        <span className="orb orbTwo" />
        <span className="orb orbThree" />
      </div>

      <header className="heroPanel">
        <div className="heroCopy">
          <span className="eyebrow">FabAgent RAG / Knowledge Studio</span>
          <h1>把文档上传、检索、问答做成一套更克制的生产级界面。</h1>
          <p>
            文件名优先展示，支持文档范围搜索，减少长路径在大量资料场景下的视觉噪音。
          </p>
        </div>

        <aside className="heroStatusPanel" aria-label="运行概览">
          <div className="runtimeStatus">
            <span className="statusDot" aria-hidden="true" />
            本地服务在线
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
              <dt>索引</dt>
              <dd>SQLite FTS5</dd>
            </div>
          </dl>
        </aside>
      </header>

      <section className="overviewRail" aria-label="状态总览">
        {overviewCards.map((card) => (
          <article className="overviewCard" key={card.label}>
            <span>{card.label}</span>
            <strong>{card.value}</strong>
            <small>{card.hint}</small>
          </article>
        ))}
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
            <strong>{lastIngest.sources.join(" · ")}</strong>
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
