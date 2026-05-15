# FabAgent RAG 评测集

这个目录记录从 `data/raw` 中挑选出的第一批评测文档和问题。目标不是一次覆盖全部资料，
而是先构造一组小而稳定的样例，用来观察解析、chunk、query plan、召回和回答是否按预期工作。

## 选取原则

- 覆盖主要文件类型：PDF、DOCX、DOC、PPT/PPTX、XLSX、MD
- 优先选择标题明确、问题容易人工核对的文件
- 不使用 `exclude__` 前缀文件
- 同时覆盖概念解释、操作步骤、故障处理、参数表查询、SPC 表格查询、总结、无答案和闲聊分流

## 选中文档

扩大测试文件规模时，优先参考 [raw_dataset_selection.md](raw_dataset_selection.md)。
该文件把 `data/raw` 中的资料分成“建议入库测试集”“暂缓入库”“建议删除或移出当前测试集”。

如果你想直接使用当前已经按成本和覆盖率筛好的核心集合，优先看
[full_test_dataset_plan.md](full_test_dataset_plan.md)。它基于 `data/raw` 的全量解析结果，
给出了一个 34 文件的 core set，以及可选的 stress set。

| 文件 | 类型 | 主要测试点 |
| --- | --- | --- |
| `data/raw/半导体术语小记.md` | MD | 概念解释、同义词、OPC 类型 |
| `data/raw/cleanroom_management.docx` | DOCX | 制度规范、巡检表、章节标题 |
| `data/raw/etch_alarm_handbook.docx` | DOCX | 报警处理、表格问答、故障排查 |
| `data/raw/mes_lot_tracking_guide.docx` | DOCX | MES 操作步骤、Hold Lot 规则 |
| `data/raw/fab_process_parameters.xlsx` | XLSX | recipe 参数查询 |
| `data/raw/spc_report.xlsx` | XLSX | SPC 上下限、异常批次 |
| `data/raw/equipment_checklist.xlsx` | XLSX | 点检表、日期和设备状态查询 |
| `data/raw/fab_alarm_log_mixed.xlsx` | XLSX | 混合报警日志、表格召回 |
| `data/raw/ICP-RIE_等离子体刻蚀机_SI_500_操作流程及使用规范.pdf` | PDF | 刻蚀 SOP、操作规程 |
| `data/raw/SQDL平台HF干法刻蚀机(HF+vapor+etcher)标准操作流程及使用规范(SOP).pdf` | PDF | HF 干法刻蚀 SOP |
| `data/raw/SE-009_自動化光阻塗佈及顯影系統-设备作业标准.pdf` | PDF | 涂胶显影设备作业标准 |
| `data/raw/電漿輔助化學氣相沉積系統(PECVD)操作手册.pdf` | PDF | PECVD 操作手册 |
| `data/raw/08微电子工艺基础光刻工艺.ppt` | PPT | 光刻培训材料、PPT 解析 |
| `data/raw/芯片生产工艺流程.doc` | DOC | 兼容旧 Office 格式、工艺流程 |
| `data/raw/段辉高-2-半导体中的材料、硅片制作流程.pptx` | PPTX | PPTX 解析、硅片制作流程 |

## JSONL 字段

`rag_eval_set.jsonl` 每行是一条评测样例：

- `id`：样例编号
- `question`：用户问题
- `intent`：期望意图，取值为 `lookup`、`summarize`、`chat`
- `top_k`：建议召回数量
- `should_retrieve`：是否应该访问知识库
- `expected_sources`：期望至少命中的来源文件
- `expected_answer_contains`：答案中建议出现的关键词
- `eval_focus`：主要观察点
- `notes`：人工 review 备注

## 使用建议

先用这些问题手动跑通链路：

```bash
rag ask "OPC有哪些类型？" --top-k 5
```

后续可以写脚本逐行读取 `rag_eval_set.jsonl`，调用 `/ask`，检查：

- intent 是否符合预期
- `query_plan.queries` 是否合理
- contexts 是否命中 `expected_sources`
- answer 是否包含 `expected_answer_contains`
- 无答案问题是否说明资料不足
- 闲聊问题是否不返回 contexts

## 自动评测

项目现在提供了一套离线评测流水线，可直接跑四层检查：

- `parse`：源文件能否稳定解析
- `chunk`：切块数量、长度分布、标题覆盖率
- `retrieval`：intent、query plan、vector / keyword / hybrid 召回效果
- `answer`：端到端回答是否命中来源、是否符合无答案/闲聊控制流

一键执行：

```bash
./scripts/eval.sh
```

常用变体：

```bash
./scripts/eval.sh --case-limit 10
./scripts/eval.sh --stages parse,chunk
./scripts/eval.sh --top-k 5
```

也可以直接使用 CLI：

```bash
rag eval
```
