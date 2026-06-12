"""
Researcher agent.

For a given sub-question (or a set of follow-up queries from the Critic), it:
  1. runs web searches,
  2. extracts clean content from the top sources,
  3. asks the LLM to pull out concrete, source-attributed facts.

The LLM never invents sources — it can only cite the numbered sources we hand it,
and every returned fact is mapped back to a real URL.
"""
from __future__ import annotations

from collections.abc import Callable

import config
from src.llm.client import LLMClient
from src.models import Fact, Source, SubQuestion
from src.tools.extract import ExtractTool
from src.tools.search import SearchTool

# on_event(event_name: str, payload: dict) -> None  (optional progress hook)
EventFn = Callable[[str, dict], None]

SYSTEM_PROMPT = """You are a meticulous research analyst. You will be given a \
QUESTION and a numbered list of SOURCES (title + extracted content).

Extract the concrete facts from the SOURCES that help answer the QUESTION. \
Rules:
- Only use information present in the provided sources. Never invent facts.
- Each fact must reference the source it came from by its integer index.
- Assign a confidence between 0 and 1 reflecting how clearly the source supports \
the fact.
- Prefer specific, verifiable statements (numbers, dates, named techniques) over \
vague generalities.
- Return at most 6 of the most relevant facts.

Respond ONLY with valid JSON in exactly this schema:
{
  "facts": [
    {"content": "the factual statement", "source_index": 0, "confidence": 0.8}
  ]
}"""


class ResearcherAgent:
    name = "researcher"

    def __init__(
        self,
        llm: LLMClient,
        search: SearchTool | None = None,
        extract: ExtractTool | None = None,
    ) -> None:
        self.llm = llm
        self.search = search or SearchTool()
        self.extract = extract or ExtractTool()
        self.tier = config.AGENT_MODEL_TIER[self.name]

    # -- public API ---------------------------------------------------------
    def research_subquestion(
        self, sq: SubQuestion, on_event: EventFn | None = None
    ) -> list[Fact]:
        queries = sq.search_queries or [sq.question]
        return self._gather(queries, sq.id, on_event)

    def research_followups(
        self, queries: list[str], qid: int = 0, on_event: EventFn | None = None
    ) -> list[Fact]:
        return self._gather(queries, qid, on_event)

    # -- internals ----------------------------------------------------------
    def _collect_sources(self, queries: list[str]) -> list[Source]:
        seen: dict[str, Source] = {}
        for q in queries:
            for src in self.search.search(q):
                if src.url not in seen:
                    seen[src.url] = src
        ranked = sorted(seen.values(), key=lambda s: s.credibility, reverse=True)
        return ranked[: config.MAX_SOURCES_PER_SUBQUESTION]

    def _gather(
        self, queries: list[str], qid: int, on_event: EventFn | None = None
    ) -> list[Fact]:
        def emit(event: str, **payload: object) -> None:
            if on_event:
                on_event(event, payload)

        emit("searching", queries=queries)
        sources = self._collect_sources(queries)
        if not sources:
            return []

        # Build numbered context, reading content for each source.
        blocks: list[str] = []
        usable: list[Source] = []
        for src in sources:
            emit("reading", source=src)
            content = self.extract.extract(src.url)
            if not content:
                continue
            idx = len(usable)
            usable.append(src)
            blocks.append(
                f"[SOURCE {idx}] {src.title} ({src.domain})\n{content}"
            )

        if not usable:
            return []

        emit("extracting", count=len(usable))
        question_text = queries[0]
        user = (
            f"QUESTION: {question_text}\n\nSOURCES:\n\n" + "\n\n---\n\n".join(blocks)
        )
        data = self.llm.complete_json(
            system=SYSTEM_PROMPT, user=user, tier=self.tier, temperature=0.2
        )

        facts: list[Fact] = []
        raw_facts = data.get("facts", []) if isinstance(data, dict) else []
        for rf in raw_facts:
            si = rf.get("source_index")
            if not isinstance(si, int) or si < 0 or si >= len(usable):
                continue
            content = (rf.get("content") or "").strip()
            if not content:
                continue
            facts.append(
                Fact(
                    content=content,
                    source=usable[si],
                    sub_question_id=qid,
                    confidence=float(rf.get("confidence", 0.5)),
                )
            )
        return facts
