"""
Planner agent.

Turns a broad user query into a small set of focused, independently-searchable
sub-questions, each with concrete search queries. This is what separates a real
research agent from a single search call: explicit decomposition + strategy.
"""
from __future__ import annotations

import config
from src.llm.client import LLMClient
from src.models import ResearchPlan, SubQuestion

SYSTEM_PROMPT = """You are an expert research strategist. You have a bias toward \
authoritative, technical sources (arXiv, Papers with Code, Hugging Face, GitHub, \
official engineering blogs) when the topic is technical or AI/ML related.

Given a user's research query, decompose it into focused sub-questions that, when \
answered together, fully address the query. Each sub-question must be independently \
researchable and paired with 1-2 precise web search queries.

Rules:
- Produce between 3 and {max_q} sub-questions. Fewer is fine for narrow queries.
- Make sub-questions specific and non-overlapping.
- Search queries should be the exact strings you would type into a search engine.
- Prefer queries that surface primary sources over listicles.

Respond ONLY with valid JSON in exactly this schema:
{{
  "interpretation": "one sentence restating what the user really wants",
  "strategy": "one sentence describing your research approach",
  "sub_questions": [
    {{
      "id": 1,
      "question": "the focused sub-question",
      "search_queries": ["query one", "query two"],
      "rationale": "why this matters to the overall query"
    }}
  ]
}}"""


class PlannerAgent:
    name = "planner"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.tier = config.AGENT_MODEL_TIER[self.name]

    def plan(self, query: str) -> ResearchPlan:
        system = SYSTEM_PROMPT.format(max_q=config.MAX_SUB_QUESTIONS)
        data = self.llm.complete_json(
            system=system,
            user=f"Research query: {query}",
            tier=self.tier,
            temperature=0.3,
        )

        raw_subs = data.get("sub_questions", []) if isinstance(data, dict) else []
        sub_questions: list[SubQuestion] = []
        for i, sq in enumerate(raw_subs[: config.MAX_SUB_QUESTIONS], start=1):
            sub_questions.append(
                SubQuestion(
                    id=sq.get("id", i),
                    question=sq.get("question", "").strip(),
                    search_queries=[q for q in sq.get("search_queries", []) if q][:2],
                    rationale=sq.get("rationale", "").strip(),
                )
            )

        # Safety net: never return an empty plan.
        if not sub_questions:
            sub_questions = [
                SubQuestion(id=1, question=query, search_queries=[query], rationale="")
            ]

        return ResearchPlan(
            original_query=query,
            interpretation=(data.get("interpretation", "") if isinstance(data, dict) else ""),
            strategy=(data.get("strategy", "") if isinstance(data, dict) else ""),
            sub_questions=sub_questions,
        )
