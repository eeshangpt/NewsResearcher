"""Pydantic output schemas for structured LLM calls.

Call sites use `model.with_structured_output(Schema)` instead of hand-rolled
JSON parsing. Empty for now — real schemas (subtopic proposals, claims,
framing labels, briefing sections, etc.) get added when the Phase 2+ agents
that need them are built.
"""
