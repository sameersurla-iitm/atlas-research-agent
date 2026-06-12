"""
Test query suite for evaluation.

A curated, category-tagged set of queries. Designed to be easy to extend — add
entries to TEST_QUERIES and the harness will pick them up automatically. Each
query notes the source domains we'd expect a strong run to surface, which makes
the results discussable in interviews ("the agent reliably found arXiv/HF").
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TestQuery:
    query: str
    category: str
    expected_domains: list[str] = field(default_factory=list)
    max_seconds: int = 60


TEST_QUERIES: list[TestQuery] = [
    TestQuery(
        "Compare LangGraph, AutoGPT, and CrewAI for building multi-agent systems",
        "technical_comparison",
        ["github.com", "langchain.com"],
    ),
    TestQuery(
        "What techniques reduce LLM inference cost in production?",
        "technical_howto",
        ["arxiv.org", "huggingface.co"],
    ),
    TestQuery(
        "How does retrieval-augmented generation (RAG) reduce hallucinations?",
        "concept_explainer",
        ["arxiv.org"],
    ),
    TestQuery(
        "What are the leading open-weight LLMs in 2025 and their benchmark scores?",
        "current_landscape",
        ["huggingface.co", "paperswithcode.com"],
    ),
    TestQuery(
        "Explain the ReAct prompting pattern and when to use it",
        "concept_explainer",
        ["arxiv.org"],
    ),
    TestQuery(
        "What is Mixture-of-Experts and why does it improve LLM efficiency?",
        "concept_explainer",
        ["arxiv.org"],
    ),
    TestQuery(
        "Best practices for evaluating RAG pipelines",
        "technical_howto",
        ["github.com", "arxiv.org"],
    ),
    TestQuery(
        "How do vector databases differ: Chroma vs Qdrant vs Weaviate?",
        "technical_comparison",
        ["github.com"],
    ),
    TestQuery(
        "What is speculative decoding and how much does it speed up inference?",
        "concept_explainer",
        ["arxiv.org"],
    ),
    TestQuery(
        "Current approaches to LLM agent memory and long-term context",
        "current_landscape",
        ["arxiv.org"],
    ),
    TestQuery(
        "Trade-offs between fine-tuning and prompt engineering for domain adaptation",
        "technical_comparison",
        ["arxiv.org"],
    ),
    TestQuery(
        "How does function calling / tool use work in modern LLMs?",
        "concept_explainer",
        ["openai.com", "anthropic.com"],
    ),
]
