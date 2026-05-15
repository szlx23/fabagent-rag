# data/raw 测试数据筛选建议

这个清单用于从 `data/raw` 中扩大测试文件规模，同时标注当前 FabAgent RAG 场景下
不适合继续保留的文件。这里不直接删除或重命名原始文件，先作为人工 review 清单。

筛选目标：

- 优先覆盖 fab/半导体制造、设备操作、SOP、报警、参数表、SPC、点检、工艺流程
- 文件可以适度扩大，但先排除明显超大、重复、金融研报/公司报告、主题偏离当前员工问答场景的资料
- 已有 `excelude__` 前缀文件继续视为排除

## 建议入库测试集

第一轮扩大测试建议选择下面这些文件。数量比初版更大，但仍以当前场景相关性为主。

| 文件 | 大小约 | 选择原因 |
| --- | ---: | --- |
| `fab_process_parameters.xlsx` | 5 KB | recipe 参数查询 |
| `fab_dirty_merged_recipe.xlsx` | 5 KB | 脏表/合并表格解析 |
| `equipment_checklist.xlsx` | 6 KB | 点检表查询 |
| `spc_report.xlsx` | 6 KB | SPC 上下限和 WARN lot |
| `fab_yield_formula_report.xlsx` | 6 KB | 良率/公式类表格 |
| `fab_complex_multisheet.xlsx` | 6 KB | 多 sheet Excel |
| `fab_pm_schedule_dirty.xlsx` | 6 KB | PM 计划表 |
| `fab_alarm_log_mixed.xlsx` | 8 KB | 混合报警日志 |
| `fab_mes_lot_tracking.xlsx` | 8 KB | MES lot 表格 |
| `fab_fdc_sensor_chart.xlsx` | 9 KB | FDC sensor 表格 |
| `半导体术语小记.md` | 11 KB | 术语、OPC、FEOL/BEOL |
| `芯片生产工艺流程.doc` | 29 KB | DOC 兼容格式、流程问答 |
| `mes_lot_tracking_guide.docx` | 36 KB | MES 操作说明 |
| `cleanroom_management.docx` | 36 KB | 洁净室制度 |
| `etch_alarm_handbook.docx` | 36 KB | 刻蚀报警处理 |
| `SE-009_自動化光阻塗佈及顯影系統-设备作业标准.pdf` | 55 KB | 小型 PDF SOP |
| `芯片制作工艺流程.doc` | 56 KB | DOC 工艺流程补充 |
| `芯片的制造过程.doc` | 96 KB | DOC 制造过程问答 |
| `倒装芯片凸点制作方法.pdf` | 258 KB | 封装/凸点工艺 |
| `半导体设备零配件清洗技术规范.pdf` | 310 KB | 设备零配件清洗规范 |
| `ArF液浸式扫描光刻机.pdf` | 410 KB | 光刻设备资料 |
| `Stepper_Training_Manual20_0.pdf` | 442 KB | Stepper 培训手册 |
| `440_litho.pdf` | 447 KB | 光刻资料 |
| `芯片制造倒装焊工艺与设备解决方案.pdf` | 453 KB | 封装设备解决方案 |
| `NSR-S622D_c.pdf` | 466 KB | 设备手册/规格 |
| `AN12522_S32K1xx_ECC错误处理_恩智浦半导体应用笔记.pdf` | 508 KB | 错误处理类文档 |
| `Microfabrication_SOPs.pdf` | 563 KB | 微加工 SOP |
| `芯片制程(以-Intel-芯片为例).docx` | 572 KB | DOCX 工艺流程 |
| `Nikon-NSR-2205-i-12-D-i-Line-Stepper.pdf` | 600 KB | Stepper 设备资料 |
| `S32K1xx_上的异常和故障检查_恩智浦半导体应用笔记.pdf` | 622 KB | 故障检查类文档 |
| `聚焦离子电子双束系统FIB培训讲义.pdf` | 948 KB | FIB 培训资料 |
| `ICP-RIE_等离子体刻蚀机_SI_500_操作流程及使用规范.pdf` | 980 KB | 刻蚀设备 SOP |
| `NES_Mini_Steppers.pdf` | 1.0 MB | Stepper 设备资料 |
| `電漿輔助化學氣相沉積系統(PECVD)操作手册.pdf` | 1.0 MB | PECVD 操作手册 |
| `06微电子工艺基础化学气相淀积.ppt` | 1.2 MB | CVD 培训材料，PPT 解析 |
| `SQDL平台HF干法刻蚀机(HF+vapor+etcher)标准操作流程及使用规范(SOP).pdf` | 1.3 MB | HF 干法刻蚀 SOP |
| `SH_系列探针台使用说明书.pdf` | 1.3 MB | 探针台说明书 |
| `图解芯片制作工艺流程.pdf` | 1.3 MB | 图解工艺流程 |
| `半導體封裝製程介紹.pdf` | 1.2 MB | 封装制程 |
| `干式真空泵概述和维护补编.pdf` | 1.7 MB | 设备维护 |
| `接触式光刻系统_MA6.pdf` | 2.2 MB | 接触式光刻设备 |
| `第四章-芯片制造概述.ppt` | 2.5 MB | PPT 工艺概述 |
| `05微电子工艺基础氧化工艺.ppt` | 2.9 MB | 氧化工艺培训 |
| `03微电子工艺基础污染控制和芯片制造基本工艺概述.ppt` | 3.1 MB | 污染控制和基本工艺 |
| `第三章-集成电路的制造工艺.ppt` | 3.4 MB | IC 制造工艺培训 |

## 暂缓入库

这些文件不是完全没用，但第一轮扩大测试暂时不建议入库。原因通常是主题偏研究/市场/公司介绍，
或者文件较大且对当前“fab 员工知识问答”帮助有限。

| 文件 | 大小约 | 暂缓原因 |
| --- | ---: | --- |
| `14-中部科學工業園區光電半導體業職業衛生危害調查與預防.pdf` | 748 KB | 职业卫生研究，可后续做安全专题 |
| `自动化系统提高晶圆研磨和抛光效率.pdf` | 1.0 MB | CMP/自动化，可后续专题 |
| `半导体设备系列_光刻机，半导体制造皇冠上的明珠.pdf` | 1.4 MB | 行业介绍偏多，保留待评估 |
| `半导体材料系列_CMP_晶圆平坦化必经之路，国产替代放量中.pdf` | 1.6 MB | 研报属性偏强，但 CMP 主题有价值 |
| `大直径半绝缘4H_SiC单晶生长及表征.pdf` | 1.7 MB | 偏材料研究，不是当前设备/SOP主线 |
| `10.1.1.526.9704.pdf` | 2.0 MB | 标题不可读，需人工确认内容 |
| `直写光刻龙头企.pdf` | 2.8 MB | 行业/公司分析属性偏强 |
| `2019-06_Plenary_Talk_EWMOVPE_Zettler_Final.pdf` | 3.3 MB | 偏会议演讲/外延专题 |
| `柔性电子制造技术基础-第4讲-PART1-2014.pdf` | 3.4 MB | 柔性电子偏离当前 fab SOP |
| `2108.11515.pdf` | 3.6 MB | 论文编号命名，需人工确认主题 |
| `750547-1.pdf` | 3.6 MB | 标题不可读，需人工确认内容 |
| `芯片制造工艺流程简介.ppt` | 4.1 MB | 可用但和多个工艺流程文档重复 |
| `段辉高-2-半导体中的材料、硅片制作流程.pptx` | 4.5 MB | 可用但较大，放第二轮 |
| `Presentation_Jan_van_Schoot_et_al.ASML.pdf` | 4.7 MB | ASML 演讲资料，偏公司/技术介绍 |
| `2020062721045275642055光刻机.pdf` | 5.2 MB | 文件名弱、可能偏行业资料 |
| `赛默飞世尔科技半导体解决方案.pdf` | 5.3 MB | 供应商解决方案，后续可作为 vendor 场景 |
| `半導體無塵室火災風險分析暨防火工程性能設計之研究.pdf` | 5.5 MB | 安全研究专题，当前可暂缓 |

## 建议删除或移出当前测试集

这些文件和当前目标场景不匹配，或者过大/重复/偏金融市场报告。建议从当前测试数据中删除，
或移动到单独的 `archive` 目录。

| 文件 | 大小约 | 建议原因 |
| --- | ---: | --- |
| `excelude__exclude__pptx__芯片微纳制造技术.pptx` | 6.2 MB | 已标记 exclude，继续删除/移出 |
| `excelude__exclude__pptx__21-系统芯片与片上通信结构.pptx` | 15 MB | 已标记 exclude，继续删除/移出 |
| `excelude__西南证券-半导体行业深度报告_复盘ASML发展历程，探寻本土光刻产业链投资机会.pdf` | 6.5 MB | 金融研报，偏投资分析 |
| `excelude__天风证券-电子制造行业报告-光刻胶.pdf` | 7.5 MB | 金融研报，偏投资分析 |
| `excelude__ASML_Annual_Report_US_GAAP_2021_unsvf2.pdf` | 9.3 MB | 公司年报，和 fab 操作知识关系弱 |
| `excelude__ASML_Government_External_Affairs_Report_2021.pdf` | 292 KB | 政府事务/公司报告，和当前场景关系弱 |
| `excelude__20210527-方正证券-方正证券电子行业深度报告_光刻胶研究框架.pdf` | 11 MB | 金融研报，token 成本高 |
| `excelude__电子工艺实训基础.pdf_by_电子工艺实训基础.pdf.pdf` | 15 MB | 文件名异常且很大，偏泛电子实训 |
| `excelude__第4章_14芯片制造概述.doc` | 16 MB | DOC 极大，和其他芯片制造概述资料重复 |
| `excelude__图解入门半导体制造工艺.pdf` | 31 MB | 超大，向量化 token 成本高 |
| `excelude__01微电子工艺基础绪论.ppt` | 9.7 MB | 大型绪论课件，信息密度不如专题文件 |
| `excelude__08微电子工艺基础光刻工艺.ppt` | 9.7 MB | 大型课件，已有更小光刻/设备资料可覆盖 |
| `excelude__09微电子工艺基础掺杂技术.ppt` | 9.0 MB | 大型课件，先移出当前测试集 |

## 建议执行方式

先只入库“建议入库测试集”。确认混合检索、metadata、Query Plan 和回答效果稳定后，
再从“暂缓入库”里按专题逐步加入。

如果要删除，建议先移动而不是直接删：

```bash
mkdir -p data/archive
mv data/raw/文件名 data/archive/
```
