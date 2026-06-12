"""
Typed data models shared across the pipeline.

Using Pydantic gives us validation, clean serialization for the UI, and a
single source of truth for the shapes that flow between agents.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    """One focused, independently-researchable facet of the user's query."""

    id: int
    question: str
    search_queries: list[str] = Field(default_factory=list)
    rationale: str = ""


class ResearchPlan(BaseModel):
    """The Planner agent's decomposition of the original query."""

    original_query: str
    interpretation: str = ""
    strategy: str = ""
    sub_questions: list[SubQuestion] = Field(default_factory=list)


class Source(BaseModel):
    """A web source the agent consulted."""

    url: str
    title: str = ""
    domain: str = ""
    credibility: float = 0.5  # 0..1, boosted for preferred technical domains


class Fact(BaseModel):
    """A single extracted claim, tied to its source and originating question."""

    content: str
    source: Source
    sub_question_id: int
    confidence: float = 0.5  # 0..1, the researcher's confidence in the claim


class KnowledgeBase(BaseModel):
    """Accumulates every fact gathered during a research run."""

    facts: list[Fact] = Field(default_factory=list)

    def add(self, fact: Fact) -> None:
        self.facts.append(fact)

    def extend(self, facts: list[Fact]) -> None:
        self.facts.extend(facts)

    def for_question(self, qid: int) -> list[Fact]:
        return [f for f in self.facts if f.sub_question_id == qid]

    def unique_sources(self) -> list[Source]:
        """Deduplicate sources by URL, keeping the highest credibility seen."""
        best: dict[str, Source] = {}
        for f in self.facts:
            existing = best.get(f.source.url)
            if existing is None or f.source.credibility > existing.credibility:
                best[f.source.url] = f.source
        return sorted(best.values(), key=lambda s: s.credibility, reverse=True)


class Critique(BaseModel):
    """The Critic agent's assessment of the current knowledge base."""

    is_sufficient: bool = False
    gaps: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)
    reasoning: str = ""


class ResearchReport(BaseModel):
    """The final synthesized deliverable."""

    query: str
    markdown: str
    sources: list[Source] = Field(default_factory=list)
    confidence_note: str = ""


class UsageStats(BaseModel):
    """Run-level telemetry surfaced in the UI for production-mindedness."""

    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    search_calls: int = 0
    pages_read: int = 0
    estimated_cost_usd: float = 0.0
    elapsed_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
