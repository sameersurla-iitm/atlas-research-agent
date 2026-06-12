"""
Research pipeline — orchestrates the four agents.

Flow:
    Planner -> Researcher (per sub-question) -> Critic (refine loop) -> Synthesizer

A `progress` callback is invoked at each stage so the UI can render live status.
The Synthesizer is exposed as a separate streaming step so the report can be
written to the screen token-by-token.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import config
from src.agents.critic import CriticAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.synthesizer import SynthesizerAgent
from src.llm.client import LLMClient
from src.models import (
    Critique,
    KnowledgeBase,
    ResearchPlan,
    ResearchReport,
    UsageStats,
)
from src.tools.extract import ExtractTool
from src.tools.search import SearchTool

# progress(stage: str, event: str, payload: object) -> None
ProgressFn = Callable[[str, str, object], None]


def _noop(stage: str, event: str, payload: object = None) -> None:  # pragma: no cover
    pass


class ResearchPipeline:
    def __init__(self, provider: str | None = None) -> None:
        self.llm = LLMClient(provider)
        # Shared tools so search/page counters aggregate across the run.
        self.search = SearchTool()
        self.extract = ExtractTool()

        self.planner = PlannerAgent(self.llm)
        self.researcher = ResearcherAgent(self.llm, self.search, self.extract)
        self.critic = CriticAgent(self.llm)
        self.synthesizer = SynthesizerAgent(self.llm)

        self._start_time: float = 0.0

    # -- stage 1-3: plan + research + critique ------------------------------
    def run_research(
        self, query: str, progress: ProgressFn = _noop
    ) -> tuple[ResearchPlan, KnowledgeBase, list[Critique]]:
        self._start_time = time.time()

        progress("planner", "start", None)
        plan = self.planner.plan(query)
        progress("planner", "done", plan)

        kb = KnowledgeBase()
        progress("researcher", "start", plan)
        total = len(plan.sub_questions)
        for i, sq in enumerate(plan.sub_questions, start=1):
            progress(
                "researcher",
                "subquestion",
                {"index": i, "total": total, "sub_question": sq},
            )

            def on_event(event: str, payload: dict) -> None:
                progress("researcher", event, payload)

            facts = self.researcher.research_subquestion(sq, on_event)
            kb.extend(facts)
            progress("researcher", "update", {"sub_question": sq, "facts": facts})
        progress("researcher", "done", kb)

        critiques: list[Critique] = []
        progress("critic", "start", None)
        for _ in range(config.MAX_CRITIC_ITERATIONS):
            critique = self.critic.review(query, plan, kb)
            critiques.append(critique)
            progress("critic", "update", critique)
            if critique.is_sufficient or not critique.follow_up_queries:
                break

            def on_event(event: str, payload: dict) -> None:
                progress("researcher", event, payload)

            extra = self.researcher.research_followups(
                critique.follow_up_queries, on_event=on_event
            )
            kb.extend(extra)
            progress("researcher", "update", {"sub_question": None, "facts": extra})
        progress("critic", "done", critiques)

        return plan, kb, critiques

    # -- stage 4: synthesis (streaming) -------------------------------------
    def synthesize_stream(
        self, query: str, plan: ResearchPlan, kb: KnowledgeBase
    ) -> Iterator[str]:
        return self.synthesizer.stream(query, plan, kb)

    def report_from_markdown(
        self, query: str, markdown: str, kb: KnowledgeBase
    ) -> ResearchReport:
        return self.synthesizer.report_from_markdown(query, markdown, kb)

    # -- telemetry ----------------------------------------------------------
    def stats(self) -> UsageStats:
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        return UsageStats(
            input_tokens=self.llm.input_tokens,
            output_tokens=self.llm.output_tokens,
            llm_calls=self.llm.calls,
            search_calls=self.search.calls,
            pages_read=self.extract.pages_read,
            estimated_cost_usd=self.llm.estimated_cost_usd(),
            elapsed_seconds=round(elapsed, 1),
        )

    # -- convenience: full run without streaming (used by evaluation) -------
    def run(self, query: str) -> tuple[ResearchReport, UsageStats]:
        plan, kb, _ = self.run_research(query)
        report = self.synthesizer.synthesize(query, plan, kb)
        return report, self.stats()
