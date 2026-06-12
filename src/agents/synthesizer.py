"""
Synthesizer agent.

Turns the gathered facts into a polished, cited Markdown report. It streams its
output token-by-token so the UI can render the report as it's written.

Citations use [n] markers that map to a numbered source list we control, so the
references are always real URLs the agent actually read.
"""
from __future__ import annotations

from collections.abc import Iterator

import config
from src.llm.client import LLMClient
from src.models import KnowledgeBase, ResearchPlan, ResearchReport, Source

SYSTEM_PROMPT = """You are an expert research writer. Using ONLY the provided \
FACTS and numbered SOURCES, write a comprehensive, well-structured report in \
Markdown that answers the QUERY.

Requirements:
- Use inline citations like [1], [2] that refer to the numbered SOURCES.
- Never cite a source number that wasn't provided.
- Structure the report with clear Markdown headings:
  ## Overview
  ## Key Findings   (use subsections where helpful)
  ## Conflicting Information   (include ONLY if conflicts exist)
  ## Conclusion
- Be precise and neutral. Prefer specifics (numbers, dates, named methods).
- Do NOT fabricate facts beyond what the sources support.
- End with a single line: "Confidence: <High|Medium|Low> — <one-sentence reason>"."""


class SynthesizerAgent:
    name = "synthesizer"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.tier = config.AGENT_MODEL_TIER[self.name]

    def _build_context(
        self, query: str, kb: KnowledgeBase
    ) -> tuple[str, list[Source]]:
        sources = kb.unique_sources()
        index = {s.url: i + 1 for i, s in enumerate(sources)}  # 1-based for citations

        source_lines = [
            f"[{index[s.url]}] {s.title or s.domain} — {s.url}" for s in sources
        ]
        fact_lines = [
            f"- {f.content}  (cite [{index[f.source.url]}], confidence {f.confidence:.1f})"
            for f in kb.facts
            if f.source.url in index
        ]

        context = (
            f"QUERY: {query}\n\n"
            f"SOURCES:\n" + "\n".join(source_lines) + "\n\n"
            f"FACTS:\n" + ("\n".join(fact_lines) or "(none)")
        )
        return context, sources

    def stream(self, query: str, plan: ResearchPlan, kb: KnowledgeBase) -> Iterator[str]:
        """Yield report tokens as they are generated."""
        context, _ = self._build_context(query, kb)
        yield from self.llm.stream(
            system=SYSTEM_PROMPT, user=context, tier=self.tier, temperature=0.4
        )

    def synthesize(self, query: str, plan: ResearchPlan, kb: KnowledgeBase) -> ResearchReport:
        """Non-streaming convenience method (used by the evaluation harness)."""
        context, sources = self._build_context(query, kb)
        markdown = self.llm.complete(
            system=SYSTEM_PROMPT, user=context, tier=self.tier, temperature=0.4
        )
        confidence_note = ""
        for line in markdown.splitlines():
            if line.strip().lower().startswith("confidence:"):
                confidence_note = line.strip()
                break
        return ResearchReport(
            query=query,
            markdown=markdown,
            sources=sources,
            confidence_note=confidence_note,
        )

    def report_from_markdown(
        self, query: str, markdown: str, kb: KnowledgeBase
    ) -> ResearchReport:
        """Wrap already-streamed markdown into a ResearchReport with sources."""
        _, sources = self._build_context(query, kb)
        confidence_note = ""
        for line in markdown.splitlines():
            if line.strip().lower().startswith("confidence:"):
                confidence_note = line.strip()
                break
        return ResearchReport(
            query=query,
            markdown=markdown,
            sources=sources,
            confidence_note=confidence_note,
        )
