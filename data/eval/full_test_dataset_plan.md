# FabAgent RAG Full Test Dataset Plan

## Corpus Snapshot

- Parsed files: `69`
- Total chunks: `3299`
- Total parsed chars: `1,445,408`
- Rough embedding tokens: `361,382`
- Estimated embedding calls at batch size `10`: `365`
- Estimated batch ingest calls: `366`
- Estimated calls per RAG question: `4`

This snapshot was generated from `data/raw` with `MINERU_MODEL_SOURCE=local`, so the PDF parser uses the locally downloaded MinerU models instead of hitting ModelScope during parsing.

## Core Test Set

The following set balances coverage and cost. It is meant to stay in the main regression corpus.

### Process / Training Slides

- `03微电子工艺基础污染控制和芯片制造基本工艺概述.ppt`
- `05微电子工艺基础氧化工艺.ppt`
- `06微电子工艺基础化学气相淀积.ppt`
- `第三章-集成电路的制造工艺.ppt`
- `芯片制造工艺流程简介.ppt`
- `段辉高-2-半导体中的材料、硅片制作流程.pptx`

### Manuals / SOPs

- `Microfabrication_SOPs.pdf`
- `ICP-RIE_等离子体刻蚀机_SI_500_操作流程及使用规范.pdf`
- `SH_系列探针台使用说明书.pdf`
- `NES_Mini_Steppers.pdf`
- `Nikon-NSR-2205-i-12-D-i-Line-Stepper.pdf`
- `接触式光刻系统_MA6.pdf`
- `2020062721045275642055光刻机.pdf`
- `半導體封裝製程介紹.pdf`
- `赛默飞世尔科技半导体解决方案.pdf`
- `干式真空泵概述和维护补编.pdf`

### Troubleshooting / Failure Analysis

- `AN12522_S32K1xx_ECC错误处理_恩智浦半导体应用笔记.pdf`
- `S32K1xx_上的异常和故障检查_恩智浦半导体应用笔记.pdf`
- `半導體無塵室火災風險分析暨防火工程性能設計之研究.pdf`

### Forms / Docs

- `cleanroom_management.docx`
- `mes_lot_tracking_guide.docx`
- `芯片制程(以-Intel-芯片为例).docx`
- `芯片制作工艺流程.doc`
- `芯片生产工艺流程.doc`
- `芯片的制造过程.doc`

### Tables / Spreadsheets

- `equipment_checklist.xlsx`
- `fab_alarm_log_mixed.xlsx`
- `fab_mes_lot_tracking.xlsx`
- `fab_fdc_sensor_chart.xlsx`
- `fab_pm_schedule_dirty.xlsx`
- `fab_complex_multisheet.xlsx`
- `fab_dirty_merged_recipe.xlsx`
- `fab_process_parameters.xlsx`
- `spc_report.xlsx`

### Lightweight Reference

- `半导体术语小记.md`

## Core Set Cost

This core set contains `34` files.

- Chunks: `914`
- Parsed chars: `372,745`
- Estimated embedding calls: `111`

That is small relative to the available quota and keeps enough variety to exercise:

- OCR-heavy PDFs
- scanned/manual PDFs
- slides and training decks
- Word documents
- multi-sheet spreadsheets
- short markdown references

## Optional Stress Set

Add these only if you want to hammer the parser or catch long-document regressions:

- `2018061811194983216511设备.pdf`
- `750547-1.pdf`

These two files dominate chunk count and OCR cost. They are useful for stress testing, but they are not necessary for a balanced regression corpus.

## Files To Drop First

If you need to shrink the corpus further, cut in this order:

1. Duplicate-looking industry reports and brochures with similar topics
2. Very large PDFs that only duplicate content already covered by the core set
3. Tiny single-purpose files that do not add new question types

## Notes

- The parse/chunk intermediates are stored under `data/eval/parse_chunk_full_local/`.
- If you want, the next step should be to turn this plan into an actual evaluation set file with question-answer pairs and retrieval checks.
