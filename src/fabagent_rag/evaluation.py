from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from time import perf_counter
import json
import re
import statistics

from fabagent_rag.chunking import Chunk
from fabagent_rag.chunking import split_text
from fabagent_rag.config import Settings
from fabagent_rag.documents import ParsedDocument, SUPPORTED_EXTENSIONS, parse_document
from fabagent_rag.intent import detect_intent
from fabagent_rag.llm import classify_intent_with_llm
from fabagent_rag.query_planner import QueryPlan, build_query_plan
from fabagent_rag.rag_service import (
    build_chunk_config,
    build_embedder,
    build_keyword_store,
    build_store,
    enrich_document_metadata,
    fused_score,
    hybrid_weights,
    merge_candidate,
    metadata_boost,
    normalize_vector_score,
    retrieval_mode,
    score_of,
)


DEFAULT_EVAL_SET = Path("data/eval/rag_eval_set.jsonl")
DEFAULT_REPORT_ROOT = Path("data/eval/reports")
DEFAULT_INTERMEDIATE_DIR = Path("data/eval/parse_chunk_full_local")
ALLOWED_STAGES = ("parse", "chunk", "retrieval", "answer")
NO_ANSWER_HINTS = (
    "资料不足",
    "没有足够信息",
    "无法确定",
    "无法判断",
    "上下文不足",
    "缺少信息",
)
CHAT_HINTS = ("FabAgent", "RAG")
RETRIEVAL_MODES = ("vector", "keyword", "hybrid")


@dataclass(frozen=True)
class EvalCase:
    """评测数据集中单条样例的结构化表示。"""

    case_id: str
    question: str
    intent: str
    top_k: int
    should_retrieve: bool
    expected_sources: list[str]
    expected_answer_contains: list[str]
    eval_focus: list[str]
    notes: str


@dataclass(frozen=True)
class ParseEvalRow:
    source: str
    status: str
    parser: str
    chars: int
    duration_ms: int
    error: str = ""


@dataclass(frozen=True)
class ChunkEvalRow:
    source: str
    chunk_count: int
    avg_chunk_chars: int
    min_chunk_chars: int
    max_chunk_chars: int
    section_title_coverage: float
    table_chunk_ratio: float
    short_chunk_ratio: float


@dataclass(frozen=True)
class RetrievalEvalRow:
    case_id: str
    expected_intent: str
    rule_intent: str
    llm_intent: str
    final_intent: str
    original_query_count: int
    planned_query_count: int
    original_source_hit: bool
    planned_vector_hit: bool
    planned_keyword_hit: bool
    planned_hybrid_hit: bool
    original_rr: float
    planned_vector_rr: float
    planned_keyword_rr: float
    planned_hybrid_rr: float
    planner_improved: bool
    error: str = ""


@dataclass(frozen=True)
class AnswerEvalRow:
    case_id: str
    expected_intent: str
    actual_intent: str
    should_retrieve: bool
    contexts_count: int
    source_hit: bool
    keyword_hit_count: int
    keyword_hit_ratio: float
    no_answer_ok: bool
    chat_ok: bool
    passed: bool
    generation_mode: str
    error: str = ""


@dataclass(frozen=True)
class EvalCaseSelection:
    cases: list[EvalCase]
    skipped_cases: list[EvalCase]
    skipped_sources: list[str]


def run_evaluation(
    settings: Settings,
    eval_set_path: Path = DEFAULT_EVAL_SET,
    stages: tuple[str, ...] = ALLOWED_STAGES,
    output_dir: Path | None = None,
    case_limit: int | None = None,
    source_limit: int | None = None,
    top_k_override: int | None = None,
    intermediate_dir: Path | None = DEFAULT_INTERMEDIATE_DIR,
) -> Path:
    """执行离线评测，并把结果写入一个新的报告目录。

    评测分四层：
    1. `parse`：检查源文件能否稳定解析。
    2. `chunk`：检查切块质量和 metadata 覆盖率。
    3. `retrieval`：比较 original query、vector、keyword、hybrid 的召回效果。
    4. `answer`：用端到端问答结果做 groundedness 和控制流检查。
    """

    raw_cases = load_eval_cases(eval_set_path)
    if case_limit is not None:
        raw_cases = raw_cases[:case_limit]
    validate_stages(stages)

    selection = select_supported_cases(raw_cases)
    cases = selection.cases
    report_dir = output_dir or default_report_dir(DEFAULT_REPORT_ROOT)
    report_dir.mkdir(parents=True, exist_ok=True)

    parsed_docs: dict[str, ParsedDocument] = {}
    sources = referenced_sources(cases)
    if source_limit is not None:
        sources = sources[:source_limit]

    summaries: dict[str, dict[str, object]] = {}
    manifest = {
        "eval_set_path": str(eval_set_path),
        "case_count": len(cases),
        "raw_case_count": len(raw_cases),
        "skipped_case_count": len(selection.skipped_cases),
        "source_count": len(sources),
        "skipped_sources": selection.skipped_sources,
        "stages": list(stages),
        "intermediate_dir": str(intermediate_dir) if intermediate_dir else "",
        "settings": build_settings_manifest(settings, top_k_override, source_limit),
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }

    if "parse" in stages:
        parse_rows, parse_summary, parsed_docs = run_parse_eval(
            settings,
            sources,
            intermediate_dir=intermediate_dir,
        )
        write_stage_rows(report_dir / "parse_rows.jsonl", parse_rows)
        summaries["parse"] = parse_summary

    if "chunk" in stages:
        cached_chunk_rows = load_intermediate_chunk_rows(
            sources,
            intermediate_dir,
            settings.min_chunk_size,
        )
        if cached_chunk_rows:
            chunk_rows, chunk_summary = summarize_chunk_rows(settings, cached_chunk_rows)
        else:
            if not parsed_docs:
                _, _, parsed_docs = run_parse_eval(
                    settings,
                    sources,
                    intermediate_dir=intermediate_dir,
                )
            chunk_rows, chunk_summary = run_chunk_eval(settings, parsed_docs)
        write_stage_rows(report_dir / "chunk_rows.jsonl", chunk_rows)
        summaries["chunk"] = chunk_summary

    if "retrieval" in stages:
        preflight_summary = run_retrieval_preflight(settings, cases)
        summaries["retrieval_preflight"] = preflight_summary
        retrieval_rows, retrieval_summary = run_retrieval_eval(
            settings,
            cases,
            top_k_override=top_k_override,
        )
        write_stage_rows(report_dir / "retrieval_rows.jsonl", retrieval_rows)
        summaries["retrieval"] = retrieval_summary

    if "answer" in stages:
        answer_rows, answer_summary = run_answer_eval(
            settings,
            cases,
            top_k_override=top_k_override,
        )
        write_stage_rows(report_dir / "answer_rows.jsonl", answer_rows)
        summaries["answer"] = answer_summary

    (report_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "README.md").write_text(
        build_report_markdown(manifest, summaries),
        encoding="utf-8",
    )
    return report_dir


def load_eval_cases(path: Path) -> list[EvalCase]:
    """读取 JSONL 评测集。"""

    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        cases.append(
            EvalCase(
                case_id=str(payload["id"]),
                question=str(payload["question"]),
                intent=str(payload["intent"]),
                top_k=int(payload.get("top_k", 4)),
                should_retrieve=bool(payload.get("should_retrieve", True)),
                expected_sources=[str(item) for item in payload.get("expected_sources", [])],
                expected_answer_contains=[
                    str(item) for item in payload.get("expected_answer_contains", [])
                ],
                eval_focus=[str(item) for item in payload.get("eval_focus", [])],
                notes=str(payload.get("notes", "")),
            )
        )
    return cases


def referenced_sources(cases: list[EvalCase]) -> list[Path]:
    """收集评测集中实际引用到的源文件。"""

    unique = sorted({source for case in cases for source in case.expected_sources})
    return [Path(source) for source in unique]


def select_supported_cases(cases: list[EvalCase]) -> EvalCaseSelection:
    """过滤当前系统已经不支持或本地不存在的评测 source。"""

    skipped_cases: list[EvalCase] = []
    skipped_sources = set()
    selected: list[EvalCase] = []
    for case in cases:
        invalid_sources = [
            source
            for source in case.expected_sources
            if not is_supported_existing_source(Path(source))
        ]
        if invalid_sources:
            skipped_cases.append(case)
            skipped_sources.update(invalid_sources)
            continue
        selected.append(case)
    return EvalCaseSelection(
        cases=selected,
        skipped_cases=skipped_cases,
        skipped_sources=sorted(skipped_sources),
    )


def is_supported_existing_source(source: Path) -> bool:
    if not source.exists():
        return False
    return source.suffix.lower() in SUPPORTED_EXTENSIONS


def parse_sources(settings: Settings, sources: list[Path]) -> dict[str, ParsedDocument]:
    """在多个阶段之间复用解析结果，避免反复跑 MinerU/Docling。"""

    parsed_docs: dict[str, ParsedDocument] = {}
    for source in sources:
        parsed_docs[str(source)] = parse_document(
            source,
            str(source),
            mineru_backend=settings.mineru_backend,
        )
    return parsed_docs


def run_parse_eval(
    settings: Settings,
    sources: list[Path],
    intermediate_dir: Path | None = DEFAULT_INTERMEDIATE_DIR,
) -> tuple[list[ParseEvalRow], dict[str, object], dict[str, ParsedDocument]]:
    rows: list[ParseEvalRow] = []
    parsed_docs: dict[str, ParsedDocument] = {}
    cached = load_intermediate_parsed_docs(sources, intermediate_dir)

    for source in sources:
        cached_document = cached.get(str(source))
        if cached_document:
            parsed_docs[str(source)] = cached_document
            rows.append(
                ParseEvalRow(
                    source=str(source),
                    status="ok",
                    parser=cached_document.metadata.get("parser", ""),
                    chars=len(cached_document.text),
                    duration_ms=0,
                )
            )
            continue

        started = perf_counter()
        try:
            document = parse_document(source, str(source), mineru_backend=settings.mineru_backend)
            duration_ms = int((perf_counter() - started) * 1000)
            parsed_docs[str(source)] = document
            rows.append(
                ParseEvalRow(
                    source=str(source),
                    status="ok",
                    parser=document.metadata.get("parser", ""),
                    chars=len(document.text),
                    duration_ms=duration_ms,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 评测需要记录所有异常
            rows.append(
                ParseEvalRow(
                    source=str(source),
                    status="error",
                    parser="",
                    chars=0,
                    duration_ms=int((perf_counter() - started) * 1000),
                    error=str(exc),
                )
            )

    ok_rows = [row for row in rows if row.status == "ok"]
    summary = {
        "source_count": len(rows),
        "success_count": len(ok_rows),
        "error_count": len(rows) - len(ok_rows),
        "empty_count": sum(1 for row in ok_rows if row.chars == 0),
        "success_rate": ratio(len(ok_rows), len(rows)),
        "avg_chars": safe_mean(row.chars for row in ok_rows),
        "avg_duration_ms": safe_mean(row.duration_ms for row in ok_rows),
        "parser_counts": dict(Counter(row.parser for row in ok_rows)),
        "cache_hit_count": sum(1 for row in ok_rows if row.duration_ms == 0),
    }
    return rows, summary, parsed_docs


def run_chunk_eval(
    settings: Settings,
    parsed_docs: dict[str, ParsedDocument],
) -> tuple[list[ChunkEvalRow], dict[str, object]]:
    rows: list[ChunkEvalRow] = []
    config = build_chunk_config(settings)

    for source, document in parsed_docs.items():
        chunks = split_text(
            document.text,
            source,
            config,
            metadata=enrich_document_metadata(document),
        )
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        section_covered = sum(1 for chunk in chunks if chunk.section_title.strip())
        table_chunks = sum(1 for chunk in chunks if chunk.content_type == "table")
        short_chunks = sum(1 for length in chunk_lengths if length < config.min_chunk_size)

        rows.append(
            ChunkEvalRow(
                source=source,
                chunk_count=len(chunks),
                avg_chunk_chars=int(safe_mean(chunk_lengths)),
                min_chunk_chars=min(chunk_lengths) if chunk_lengths else 0,
                max_chunk_chars=max(chunk_lengths) if chunk_lengths else 0,
                section_title_coverage=ratio(section_covered, len(chunks)),
                table_chunk_ratio=ratio(table_chunks, len(chunks)),
                short_chunk_ratio=ratio(short_chunks, len(chunks)),
            )
        )

    summary = {
        "source_count": len(rows),
        "avg_chunk_count": safe_mean(row.chunk_count for row in rows),
        "avg_chunk_chars": safe_mean(row.avg_chunk_chars for row in rows),
        "avg_section_title_coverage": safe_mean(row.section_title_coverage for row in rows),
        "avg_table_chunk_ratio": safe_mean(row.table_chunk_ratio for row in rows),
        "avg_short_chunk_ratio": safe_mean(row.short_chunk_ratio for row in rows),
        "config": asdict(config),
    }
    return rows, summary


def summarize_chunk_rows(
    settings: Settings,
    rows: list[ChunkEvalRow],
) -> tuple[list[ChunkEvalRow], dict[str, object]]:
    summary = {
        "source_count": len(rows),
        "avg_chunk_count": safe_mean(row.chunk_count for row in rows),
        "avg_chunk_chars": safe_mean(row.avg_chunk_chars for row in rows),
        "avg_section_title_coverage": safe_mean(row.section_title_coverage for row in rows),
        "avg_table_chunk_ratio": safe_mean(row.table_chunk_ratio for row in rows),
        "avg_short_chunk_ratio": safe_mean(row.short_chunk_ratio for row in rows),
        "config": asdict(build_chunk_config(settings)),
        "cache_hit_count": len(rows),
    }
    return rows, summary


def run_retrieval_eval(
    settings: Settings,
    cases: list[EvalCase],
    top_k_override: int | None = None,
) -> tuple[list[RetrievalEvalRow], dict[str, object]]:
    rows: list[RetrievalEvalRow] = []
    retrieval_cases = [
        case for case in cases if case.should_retrieve and case.expected_sources and case.intent != "chat"
    ]

    for case in retrieval_cases:
        top_k = top_k_override or case.top_k
        rule_intent = detect_intent(case.question)
        llm_intent = classify_intent_with_llm(
            case.question,
            settings.inference_api_key,
            settings.inference_base_url,
            settings.inference_model,
        )
        final_intent = llm_intent or rule_intent

        try:
            query_plan = build_query_plan(
                case.question,
                final_intent,
                settings.inference_api_key,
                settings.inference_base_url,
                settings.inference_model,
            )
            original_plan = QueryPlan(
                original_query=case.question,
                rewritten_query=case.question,
                expanded_queries=[],
            )

            original_contexts = search_contexts_for_eval(settings, original_plan, top_k, final_intent, "hybrid")
            planned_vector = search_contexts_for_eval(settings, query_plan, top_k, final_intent, "vector")
            planned_keyword = search_contexts_for_eval(settings, query_plan, top_k, final_intent, "keyword")
            planned_hybrid = search_contexts_for_eval(settings, query_plan, top_k, final_intent, "hybrid")

            original_rr = reciprocal_rank(original_contexts, case.expected_sources)
            planned_hybrid_rr = reciprocal_rank(planned_hybrid, case.expected_sources)
            rows.append(
                RetrievalEvalRow(
                    case_id=case.case_id,
                    expected_intent=case.intent,
                    rule_intent=rule_intent,
                    llm_intent=llm_intent or "",
                    final_intent=final_intent,
                    original_query_count=len(original_plan.queries()),
                    planned_query_count=len(query_plan.queries()),
                    original_source_hit=source_hit(original_contexts, case.expected_sources),
                    planned_vector_hit=source_hit(planned_vector, case.expected_sources),
                    planned_keyword_hit=source_hit(planned_keyword, case.expected_sources),
                    planned_hybrid_hit=source_hit(planned_hybrid, case.expected_sources),
                    original_rr=original_rr,
                    planned_vector_rr=reciprocal_rank(planned_vector, case.expected_sources),
                    planned_keyword_rr=reciprocal_rank(planned_keyword, case.expected_sources),
                    planned_hybrid_rr=planned_hybrid_rr,
                    planner_improved=planned_hybrid_rr > original_rr,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 评测阶段不能因为单题失败而中断
            rows.append(
                RetrievalEvalRow(
                    case_id=case.case_id,
                    expected_intent=case.intent,
                    rule_intent=rule_intent,
                    llm_intent=llm_intent or "",
                    final_intent=final_intent,
                    original_query_count=1,
                    planned_query_count=0,
                    original_source_hit=False,
                    planned_vector_hit=False,
                    planned_keyword_hit=False,
                    planned_hybrid_hit=False,
                    original_rr=0.0,
                    planned_vector_rr=0.0,
                    planned_keyword_rr=0.0,
                    planned_hybrid_rr=0.0,
                    planner_improved=False,
                    error=str(exc),
                )
            )

    ok_rows = [row for row in rows if not row.error]
    summary = {
        "case_count": len(rows),
        "error_count": len(rows) - len(ok_rows),
        "intent_accuracy": ratio(
            sum(1 for row in ok_rows if row.final_intent == row.expected_intent),
            len(ok_rows),
        ),
        "rule_intent_accuracy": ratio(
            sum(1 for row in ok_rows if row.rule_intent == row.expected_intent),
            len(ok_rows),
        ),
        "llm_intent_coverage": ratio(sum(1 for row in ok_rows if row.llm_intent), len(ok_rows)),
        "original_hit_rate": ratio(sum(1 for row in ok_rows if row.original_source_hit), len(ok_rows)),
        "vector_hit_rate": ratio(sum(1 for row in ok_rows if row.planned_vector_hit), len(ok_rows)),
        "keyword_hit_rate": ratio(sum(1 for row in ok_rows if row.planned_keyword_hit), len(ok_rows)),
        "hybrid_hit_rate": ratio(sum(1 for row in ok_rows if row.planned_hybrid_hit), len(ok_rows)),
        "original_mrr": safe_mean(row.original_rr for row in ok_rows),
        "vector_mrr": safe_mean(row.planned_vector_rr for row in ok_rows),
        "keyword_mrr": safe_mean(row.planned_keyword_rr for row in ok_rows),
        "hybrid_mrr": safe_mean(row.planned_hybrid_rr for row in ok_rows),
        "planner_improvement_rate": ratio(
            sum(1 for row in ok_rows if row.planner_improved),
            len(ok_rows),
        ),
    }
    return rows, summary


def run_answer_eval(
    settings: Settings,
    cases: list[EvalCase],
    top_k_override: int | None = None,
) -> tuple[list[AnswerEvalRow], dict[str, object]]:
    rows: list[AnswerEvalRow] = []
    generation_mode = "llm" if settings.inference_api_key and settings.inference_model else "fallback"

    from fabagent_rag.rag_service import answer_question

    for case in cases:
        top_k = top_k_override or case.top_k
        try:
            result = answer_question(settings, case.question, top_k)
            contexts = result.get("contexts", [])
            answer = str(result.get("answer") or "")
            actual_intent = str(result.get("intent") or "")
            keyword_hit_count = matched_keyword_count(answer, case.expected_answer_contains)
            keyword_ratio = ratio(keyword_hit_count, len(case.expected_answer_contains))
            answer_has_no_answer_hint = contains_any(answer, NO_ANSWER_HINTS)
            answer_has_chat_hint = contains_any(answer, case.expected_answer_contains or CHAT_HINTS)
            hit = source_hit(contexts, case.expected_sources)
            no_answer_case = case.should_retrieve and not case.expected_sources

            passed = evaluate_answer_case(
                case,
                actual_intent=actual_intent,
                contexts=contexts,
                source_hit_value=hit,
                keyword_hit_ratio=keyword_ratio,
                no_answer_ok=no_answer_case and answer_has_no_answer_hint,
                chat_ok=(case.intent == "chat" and not contexts and answer_has_chat_hint),
            )

            rows.append(
                AnswerEvalRow(
                    case_id=case.case_id,
                    expected_intent=case.intent,
                    actual_intent=actual_intent,
                    should_retrieve=case.should_retrieve,
                    contexts_count=len(contexts),
                    source_hit=hit,
                    keyword_hit_count=keyword_hit_count,
                    keyword_hit_ratio=keyword_ratio,
                    no_answer_ok=no_answer_case and answer_has_no_answer_hint,
                    chat_ok=(case.intent == "chat" and not contexts and answer_has_chat_hint),
                    passed=passed,
                    generation_mode=generation_mode,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 评测阶段不能因为单题失败而中断
            rows.append(
                AnswerEvalRow(
                    case_id=case.case_id,
                    expected_intent=case.intent,
                    actual_intent="",
                    should_retrieve=case.should_retrieve,
                    contexts_count=0,
                    source_hit=False,
                    keyword_hit_count=0,
                    keyword_hit_ratio=0.0,
                    no_answer_ok=False,
                    chat_ok=False,
                    passed=False,
                    generation_mode=generation_mode,
                    error=str(exc),
                )
            )

    ok_rows = [row for row in rows if not row.error]
    cases_by_id = {case.case_id: case for case in cases}
    source_rows = [
        row
        for row in ok_rows
        if cases_by_id.get(row.case_id)
        and cases_by_id[row.case_id].should_retrieve
        and cases_by_id[row.case_id].expected_sources
    ]
    no_answer_cases = [case for case in cases if case.should_retrieve and not case.expected_sources]
    chat_cases = [case for case in cases if case.intent == "chat"]
    summary = {
        "case_count": len(rows),
        "error_count": len(rows) - len(ok_rows),
        "pass_rate": ratio(sum(1 for row in ok_rows if row.passed), len(ok_rows)),
        "intent_accuracy": ratio(
            sum(1 for row in ok_rows if row.actual_intent == row.expected_intent),
            len(ok_rows),
        ),
        "source_hit_rate": ratio(sum(1 for row in source_rows if row.source_hit), len(source_rows)),
        "avg_keyword_hit_ratio": safe_mean(row.keyword_hit_ratio for row in ok_rows),
        "no_answer_pass_rate": ratio(
            sum(1 for row in ok_rows if cases_by_id[row.case_id] in no_answer_cases and row.no_answer_ok),
            len(no_answer_cases),
        ),
        "chat_pass_rate": ratio(
            sum(1 for row in ok_rows if row.expected_intent == "chat" and row.chat_ok),
            len(chat_cases),
        ),
        "generation_mode": generation_mode,
    }
    return rows, summary


def run_retrieval_preflight(settings: Settings, cases: list[EvalCase]) -> dict[str, object]:
    expected_sources = sorted({source for case in cases for source in case.expected_sources})
    if not expected_sources:
        return {"expected_source_count": 0, "indexed_source_count": 0, "missing_sources": []}

    keyword_store = build_keyword_store(settings)
    indexed_sources = {str(row.get("source") or "") for row in keyword_store.list_documents()}
    missing_sources = [source for source in expected_sources if source not in indexed_sources]
    return {
        "expected_source_count": len(expected_sources),
        "indexed_source_count": len(expected_sources) - len(missing_sources),
        "missing_source_count": len(missing_sources),
        "missing_sources": missing_sources,
    }


def search_contexts_for_eval(
    settings: Settings,
    query_plan: QueryPlan,
    top_k: int,
    intent: str,
    mode: str,
) -> list[dict[str, object]]:
    """按指定检索模式执行同一组 query，用于比较模块效果。"""

    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"未知检索模式：{mode}")

    embedder = build_embedder(settings)
    store = build_store(settings, embedder.dimension)
    keyword_store = build_keyword_store(settings)
    queries = query_plan.queries()
    embeddings = embedder.encode(queries) if mode in {"vector", "hybrid"} else []
    weights = hybrid_weights(settings, intent) if mode == "hybrid" else (1.0, 0.0)

    candidates: dict[tuple[object, object, object, object, object], dict[str, object]] = {}
    for index, query in enumerate(queries):
        if mode in {"vector", "hybrid"}:
            for context in store.search(embeddings[index], top_k=top_k * 2):
                merge_candidate(
                    candidates,
                    {
                        **context,
                        "matched_query": query,
                        "vector_score": normalize_vector_score(context.get("score")),
                    },
                )

        if mode in {"keyword", "hybrid"}:
            for context in keyword_store.search(query, top_k=top_k * 2):
                merge_candidate(
                    candidates,
                    {
                        **context,
                        "matched_query": query,
                        "keyword_score": float(context.get("keyword_score") or 0.0),
                    },
                )

    for context in candidates.values():
        if mode == "hybrid":
            context["metadata_boost"] = metadata_boost(context, queries)
            context["score"] = fused_score(context, weights)
        elif mode == "vector":
            context["metadata_boost"] = 0.0
            context["score"] = float(context.get("vector_score") or 0.0)
        else:
            context["metadata_boost"] = 0.0
            context["score"] = float(context.get("keyword_score") or 0.0)
        context["retrieval_mode"] = retrieval_mode(context)

    return sorted(candidates.values(), key=score_of, reverse=True)[:top_k]


def reciprocal_rank(contexts: list[dict[str, object]], expected_sources: list[str]) -> float:
    """计算第一条相关结果的倒数排名。"""

    expected = {normalize_text(source) for source in expected_sources}
    if not expected:
        return 0.0

    for index, context in enumerate(contexts, start=1):
        if normalize_text(str(context.get("source") or "")) in expected:
            return 1.0 / index
    return 0.0


def source_hit(contexts: list[dict[str, object]], expected_sources: list[str]) -> bool:
    """判断 top-k 内是否命中任一目标来源。"""

    expected = {normalize_text(source) for source in expected_sources}
    return any(normalize_text(str(context.get("source") or "")) in expected for context in contexts)


def matched_keyword_count(answer: str, keywords: list[str]) -> int:
    """统计答案命中了多少个期望关键词。"""

    normalized_answer = normalize_text(answer)
    return sum(1 for keyword in keywords if normalize_text(keyword) in normalized_answer)


def evaluate_answer_case(
    case: EvalCase,
    actual_intent: str,
    contexts: list[dict[str, object]],
    source_hit_value: bool,
    keyword_hit_ratio: float,
    no_answer_ok: bool,
    chat_ok: bool,
) -> bool:
    """给单题一个保守的规则分。

    当前项目还没有单独的 LLM judge，所以这里只做“是否明显过关”的启发式判断：
    - `chat`：不检索且回答看起来是聊天回复。
    - `no-answer`：说明资料不足。
    - 其他检索题：命中目标 source，并且答案至少覆盖部分关键点。
    """

    if case.intent == "chat":
        return chat_ok and actual_intent == "chat"

    if case.should_retrieve and not case.expected_sources:
        return actual_intent != "chat" and no_answer_ok

    if not case.should_retrieve:
        return not contexts

    required_keyword_ratio = 0.5 if case.expected_answer_contains else 0.0
    return (
        actual_intent == case.intent
        and source_hit_value
        and keyword_hit_ratio >= required_keyword_ratio
    )


def write_stage_rows(path: Path, rows: list[object]) -> None:
    """统一写 JSON 行，便于后续脚本分析。"""

    path.write_text(
        "\n".join(json.dumps(asdict(row), ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def build_report_markdown(manifest: dict[str, object], summaries: dict[str, dict[str, object]]) -> str:
    """生成一份适合人工 review 的 Markdown 报告。"""

    lines = [
        "# FabAgent RAG Evaluation Report",
        "",
        "## Manifest",
        "",
        f"- Eval set: `{manifest['eval_set_path']}`",
        f"- Cases: `{manifest['case_count']}`",
        f"- Skipped cases: `{manifest.get('skipped_case_count', 0)}`",
        f"- Sources: `{manifest['source_count']}`",
        f"- Stages: `{', '.join(manifest['stages'])}`",
        f"- Intermediate dir: `{manifest.get('intermediate_dir') or 'disabled'}`",
        f"- Created at: `{manifest['created_at']}`",
    ]

    for stage_name, summary in summaries.items():
        lines.extend(
            [
                "",
                f"## {stage_name.title()}",
                "",
                "```json",
                json.dumps(summary, ensure_ascii=False, indent=2),
                "```",
            ]
        )

    stage_files = {
        "parse": "`parse_rows.jsonl`: 解析逐文件结果",
        "chunk": "`chunk_rows.jsonl`: 切块逐文件结果",
        "retrieval": "`retrieval_rows.jsonl`: 检索逐题结果",
        "answer": "`answer_rows.jsonl`: 问答逐题结果",
    }
    lines.extend(["", "## Files", "", "- `summary.json`: 汇总指标"])
    for stage_name in manifest["stages"]:
        entry = stage_files.get(str(stage_name))
        if entry:
            lines.append(f"- {entry}")
    return "\n".join(lines) + "\n"


def default_report_dir(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return root / timestamp


def safe_mean(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(float(statistics.mean(items)), 4)


def ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def contains_any(text: str, needles: tuple[str, ...] | list[str]) -> bool:
    normalized_text = normalize_text(text)
    return any(normalize_text(needle) in normalized_text for needle in needles if needle)


def validate_stages(stages: tuple[str, ...]) -> None:
    """拒绝未知阶段名，避免把拼写错误静默当成空跑。"""

    invalid = sorted(set(stages) - set(ALLOWED_STAGES))
    if invalid:
        invalid_names = ", ".join(invalid)
        allowed_names = ", ".join(ALLOWED_STAGES)
        raise ValueError(f"未知评测阶段：{invalid_names}。可选值：{allowed_names}")


def load_intermediate_parsed_docs(
    sources: list[Path],
    intermediate_dir: Path | None,
) -> dict[str, ParsedDocument]:
    if not intermediate_dir or not intermediate_dir.exists():
        return {}

    file_rows = load_intermediate_file_rows(intermediate_dir)
    parsed_docs: dict[str, ParsedDocument] = {}
    for source in sources:
        row = file_rows.get(str(source))
        parsed_path = intermediate_dir / "parsed" / f"{safe_file_stem(source)}.md"
        if not row or row.get("status") != "ok" or not parsed_path.exists():
            continue
        parsed_docs[str(source)] = ParsedDocument(
            source=str(source),
            text=parsed_path.read_text(encoding="utf-8"),
            metadata={
                "parser": str(row.get("parser") or ""),
                "file_ext": source.suffix.lower(),
            },
        )
    return parsed_docs


def load_intermediate_chunk_rows(
    sources: list[Path],
    intermediate_dir: Path | None,
    min_chunk_size: int,
) -> list[ChunkEvalRow]:
    if not intermediate_dir or not intermediate_dir.exists():
        return []

    file_rows = load_intermediate_file_rows(intermediate_dir)
    rows: list[ChunkEvalRow] = []
    for source in sources:
        row = file_rows.get(str(source))
        chunk_path = intermediate_dir / "chunks" / f"{safe_file_stem(source)}.jsonl"
        if not row or row.get("status") != "ok" or not chunk_path.exists():
            continue

        chunks = load_chunk_jsonl(chunk_path)
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        section_covered = sum(1 for chunk in chunks if chunk.section_title.strip())
        table_chunks = sum(1 for chunk in chunks if chunk.content_type == "table")
        rows.append(
            ChunkEvalRow(
                source=str(source),
                chunk_count=len(chunks),
                avg_chunk_chars=int(safe_mean(chunk_lengths)),
                min_chunk_chars=min(chunk_lengths) if chunk_lengths else 0,
                max_chunk_chars=max(chunk_lengths) if chunk_lengths else 0,
                section_title_coverage=ratio(section_covered, len(chunks)),
                table_chunk_ratio=ratio(table_chunks, len(chunks)),
                short_chunk_ratio=ratio(
                    sum(1 for length in chunk_lengths if min_chunk_size and length < min_chunk_size),
                    len(chunks),
                ),
            )
        )
    return rows if len(rows) == len(sources) else []


def load_intermediate_file_rows(intermediate_dir: Path) -> dict[str, dict[str, object]]:
    summary_path = intermediate_dir / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return {str(row.get("source")): row for row in payload.get("files", [])}
    return {}


def load_chunk_jsonl(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        chunks.append(
            Chunk(
                text=str(payload.get("text") or ""),
                source=str(payload.get("source") or ""),
                index=int(payload.get("index") or 0),
                page=payload.get("page") if isinstance(payload.get("page"), int) else None,
                section_title=str(payload.get("section_title") or ""),
                file_ext=str(payload.get("file_ext") or ""),
                content_type=str(payload.get("content_type") or "text"),
                sheet_name=str(payload.get("sheet_name") or ""),
                parser=str(payload.get("parser") or ""),
                chunk_id=str(payload.get("chunk_id") or ""),
                ingested_at=str(payload.get("ingested_at") or ""),
            )
        )
    return chunks


def safe_file_stem(path: Path) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._-") or "document"
    digest = sha1(path.name.encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{digest}"


def build_settings_manifest(
    settings: Settings,
    top_k_override: int | None,
    source_limit: int | None,
) -> dict[str, object]:
    return {
        "milvus_collection": settings.milvus_collection,
        "embedding_model": settings.embedding_model,
        "inference_model": settings.inference_model,
        "keyword_index_path": settings.keyword_index_path,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "min_chunk_size": settings.min_chunk_size,
        "mineru_backend": settings.mineru_backend,
        "top_k_override": top_k_override,
        "source_limit": source_limit,
    }
