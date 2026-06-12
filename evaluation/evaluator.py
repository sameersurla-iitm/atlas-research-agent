"""
Evaluation runner.

Three complementary signals:
  1. Citation quality  — objective, free, computed from the report text.
  2. LLM-as-judge       — accuracy / completeness / coherence (1-10).
  3. Speed              — wall-clock seconds per query.

Run from the project root:

    python -m evaluation.evaluator            # full suite
    python -m evaluation.evaluator --limit 3  # quick smoke run

Results are printed and written to evaluation/results/eval_<timestamp>.md.
"""
from __future__ import annotations

import argparse
import re
import statistics
from datetime import datetime
from pathlib import Path

import config
from src.llm.client import LLMClient
from src.models import ResearchReport
from src.pipeline import ResearchPipeline
from evaluation.test_queries import TEST_QUERIES, TestQuery

RESULTS_DIR = Path(__file__).parent / "results"

JUDGE_SYSTEM = """You are a strict, fair research-report evaluator. Given a QUERY \
and a REPORT, score the report from 1-10 on each dimension:
- accuracy: are the claims correct and well-supported by citations?
- completeness: does it cover the important facets of the query?
- coherence: is it well-structured and clearly written?
- citation_quality: are citations present, specific, and well-placed?

Respond ONLY with valid JSON:
{"accuracy": 0, "completeness": 0, "coherence": 0, "citation_quality": 0,
 "overall": 0, "reasoning": "one or two sentences"}"""


def evaluate_citations(report: ResearchReport) -> dict:
    """Objective, network-free citation metrics."""
    md = report.markdown or ""
    markers = re.findall(r"\[(\d+)\]", md)
    cited = {int(m) for m in markers}
    n_sources = len(report.sources)
    valid = {c for c in cited if 1 <= c <= n_sources}
    invalid = {c for c in cited if c < 1 or c > n_sources}
    words = max(len(md.split()), 1)
    return {
        "num_sources": n_sources,
        "citation_markers": len(markers),
        "unique_sources_cited": len(valid),
        "invalid_citations": len(invalid),
        "citation_density_per_100w": round(len(markers) / words * 100, 2),
        "source_coverage": round(len(valid) / n_sources, 2) if n_sources else 0.0,
    }


def llm_judge(judge: LLMClient, query: str, report: ResearchReport) -> dict:
    user = (
        f"QUERY: {query}\n\nREPORT:\n{report.markdown}\n\n"
        f"(The report cited {len(report.sources)} sources.)"
    )
    try:
        data = judge.complete_json(JUDGE_SYSTEM, user, temperature=0.0)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    if not isinstance(data, dict):
        return {"error": "judge returned non-object"}
    return data


def run_one(tq: TestQuery, judge: LLMClient) -> dict:
    pipeline = ResearchPipeline()  # fresh state => clean per-query telemetry
    report, stats = pipeline.run(tq.query)
    citations = evaluate_citations(report)
    judged = llm_judge(judge, tq.query, report)
    return {
        "query": tq.query,
        "category": tq.category,
        "seconds": stats.elapsed_seconds,
        "within_time": stats.elapsed_seconds <= tq.max_seconds,
        "tokens": stats.total_tokens,
        "est_cost_usd": stats.estimated_cost_usd,
        "citations": citations,
        "judge": judged,
    }


def _avg(values: list[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(vals), 2) if vals else 0.0


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "avg_seconds": _avg([r["seconds"] for r in rows]),
        "avg_sources": _avg([r["citations"]["num_sources"] for r in rows]),
        "avg_source_coverage": _avg([r["citations"]["source_coverage"] for r in rows]),
        "avg_invalid_citations": _avg([r["citations"]["invalid_citations"] for r in rows]),
        "avg_accuracy": _avg([r["judge"].get("accuracy", 0) for r in rows]),
        "avg_completeness": _avg([r["judge"].get("completeness", 0) for r in rows]),
        "avg_coherence": _avg([r["judge"].get("coherence", 0) for r in rows]),
        "avg_overall": _avg([r["judge"].get("overall", 0) for r in rows]),
        "avg_cost_usd": round(_avg([r["est_cost_usd"] for r in rows]), 4),
    }


def write_report(rows: list[dict], summary: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"eval_{ts}.md"

    lines = [
        f"# Evaluation Report — {ts}",
        "",
        f"- Queries: **{summary['n']}**",
        f"- Avg response time: **{summary['avg_seconds']}s**",
        f"- Avg sources/query: **{summary['avg_sources']}**",
        f"- Avg source coverage: **{summary['avg_source_coverage']}**",
        f"- Avg invalid citations: **{summary['avg_invalid_citations']}**",
        f"- Avg accuracy (judge): **{summary['avg_accuracy']}/10**",
        f"- Avg completeness (judge): **{summary['avg_completeness']}/10**",
        f"- Avg coherence (judge): **{summary['avg_coherence']}/10**",
        f"- Avg overall (judge): **{summary['avg_overall']}/10**",
        f"- Avg equivalent cost/query: **${summary['avg_cost_usd']}**",
        "",
        "## Per-query results",
        "",
        "| Query | Cat | Sec | Src | Cov | Acc | Cmp | Coh | Overall |",
        "|-------|-----|-----|-----|-----|-----|-----|-----|---------|",
    ]
    for r in rows:
        j = r["judge"]
        c = r["citations"]
        q = r["query"][:48] + ("…" if len(r["query"]) > 48 else "")
        lines.append(
            f"| {q} | {r['category']} | {r['seconds']} | {c['num_sources']} | "
            f"{c['source_coverage']} | {j.get('accuracy','-')} | "
            f"{j.get('completeness','-')} | {j.get('coherence','-')} | "
            f"{j.get('overall','-')} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the research agent.")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N queries.")
    args = parser.parse_args()

    if not config.has_llm_key():
        raise SystemExit(
            f"No API key for provider '{config.LLM_PROVIDER}'. "
            f"Set it in .env before running the evaluation."
        )

    queries = TEST_QUERIES[: args.limit] if args.limit else TEST_QUERIES
    judge = LLMClient()  # judge uses the same provider's strong model

    rows: list[dict] = []
    for i, tq in enumerate(queries, start=1):
        print(f"[{i}/{len(queries)}] {tq.query}")
        try:
            rows.append(run_one(tq, judge))
        except Exception as exc:  # noqa: BLE001 - keep going on individual failures
            print(f"   ! failed: {exc}")

    if not rows:
        raise SystemExit("No successful runs to summarize.")

    summary = summarize(rows)
    path = write_report(rows, summary)
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>22}: {v}")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
