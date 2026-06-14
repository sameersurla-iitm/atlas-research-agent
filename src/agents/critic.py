"""
Critic agent.

Reviews the knowledge base against the original query and plan, then decides
whether the research is good enough. If not, it names the gaps/conflicts and
proposes concrete follow-up search queries — this is the self-correction loop
that makes the system "agentic" rather than a one-shot pipeline.
"""
from __future__ import annotations

import config
from src.llm.client import LLMClient
from src.models import Critique, KnowledgeBase, ResearchPlan

SYSTEM_PROMPT = """You are a rigorous research critic. You will receive the \
original QUERY, the list of SUB-QUESTIONS, and the FACTS gathered so far.

Assess the research:
- Coverage: is every sub-question meaningfully answered?
- Gaps: which important aspects are unanswered or thin?
- Conflicts: do any facts contradict each other?
- Sufficiency: is this enough to write a confident, well-rounded report?

If the research is NOT sufficient, propose up to 3 specific follow-up search \
queries that would close the most important gaps. If it IS sufficient, return an \
empty follow-up list.

IMPORTANT: Keep EVERY string value SHORT (under 15 words). No examples, no \
brackets, no parentheses inside the strings. The entire JSON must fit in \
300 tokens.

Respond ONLY with valid JSON in exactly this schema:
{
  "is_sufficient": true,
  "gaps": ["short gap description"],
  "conflicts": ["short conflict description"],
  "follow_up_queries": ["short search query"],
  "reasoning": "one concise sentence"
}"""


class CriticAgent:
    name = "critic"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.tier = config.AGENT_MODEL_TIER[self.name]

    def review(self, query: str, plan: ResearchPlan, kb: KnowledgeBase) -> Critique:
        sub_q_text = "\n".join(f"- ({sq.id}) {sq.question}" for sq in plan.sub_questions)
        facts_text = "\n".join(
            f"- [{f.confidence:.1f}] {f.content}  (source: {f.source.domain})"
            for f in kb.facts
        ) or "(no facts gathered)"

        user = (
            f"QUERY: {query}\n\nSUB-QUESTIONS:\n{sub_q_text}\n\n"
            f"FACTS ({len(kb.facts)}):\n{facts_text}"
        )
        # The Critic is an optional refinement step. If its response can't be
        # parsed (e.g. a provider truncates the JSON), we must NOT crash the whole
        # run — we default to "sufficient" so the pipeline proceeds to synthesis
        # and the user still gets a complete, cited report.
        try:
            data = self.llm.complete_json(
                system=SYSTEM_PROMPT, user=user, tier=self.tier,
                temperature=0.2, max_tokens=512,
            )
        except Exception:  # noqa: BLE001 - graceful degradation, never fatal
            return Critique(
                is_sufficient=True,
                reasoning="Critic step skipped (response unavailable); research accepted as-is.",
            )

        if not isinstance(data, dict):
            return Critique(is_sufficient=True, reasoning="Critique unavailable.")

        return Critique(
            is_sufficient=bool(data.get("is_sufficient", True)),
            gaps=[g for g in data.get("gaps", []) if g][:5],
            conflicts=[c for c in data.get("conflicts", []) if c][:5],
            follow_up_queries=[q for q in data.get("follow_up_queries", []) if q][:3],
            reasoning=data.get("reasoning", "").strip(),
        )
