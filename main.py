"""Nova Notebook — Personal Research Notebook Assistant.

Scenario
--------
Nova Notebook answers research questions grounded in a set of saved notes,
remembers each notebook across restarts, plans multi-part questions before
answering, and checks its own draft against retrieved evidence before it is
returned.

Your task
---------
This question has two parts.

1. Write three prompts from scratch in prompts/ (only guidance comments are
   provided): plan_prompt.txt, answer_prompt.txt, reflect_prompt.txt.
2. Complete four TODO functions in this file:
   1. resume_or_start_session
   2. route_after_plan
   3. extract_citations
   4. reflect_and_maybe_revise

Everything else — compaction, session storage, the local retriever, and the
run_notebook_query orchestrator below — is supplied. Do not modify
utils/memory.py, utils/storage.py, utils/retrieval.py, utils/state.py, or
the orchestrator/LLM-wiring code in this file.
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from utils.memory import compact_history
from utils.retrieval import build_index, retrieve_top_k
from utils.state import AgentState
from utils.storage import load_session, save_session

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"
CITATION_PATTERN = re.compile(r"\[(NOTE-[A-Za-z0-9_-]+)\]")


def resume_or_start_session(
    notebook_id: str,
    storage_dir: str,
    keep_recent: int,
    summarize_fn: Callable[[list[dict[str, Any]]], str],
) -> AgentState:
    """Load a notebook's saved state, compacting it if needed, or start fresh.

    TODO 1 contract:
    1. Call load_session(storage_dir, notebook_id).
    2. If nothing is stored yet (None is returned), return a fresh
       AgentState with this notebook_id, an empty messages list, an empty
       plan list, an empty retrieved list, and empty-string draft_answer /
       final_answer, and an empty citations list.
    3. If a prior session IS found and its stored "messages" list has more
       than keep_recent entries, replace its "messages" with the result of
       compact_history(messages, keep_recent, summarize_fn) before
       returning it.
    4. If the prior session has keep_recent or fewer messages, return the
       loaded state unchanged (still compacted-shape-compatible; just don't
       call compact_history).
    5. Either way, the returned state's "notebook_id" must equal the
       notebook_id argument.
    """
    state = load_session(storage_dir, notebook_id)
    if state is None:
        return AgentState(
            notebook_id=notebook_id,
            messages=[],
            plan=[],
            retrieved=[],
            draft_answer="",
            citations=[],
            final_answer="",
        )

    messages = state.get("messages", [])
    if len(messages) > keep_recent:
        state["messages"] = compact_history(messages, keep_recent, summarize_fn)
    state["notebook_id"] = notebook_id
    return state


def route_after_plan(plan_steps: list[str]) -> str:
    """Decide whether the plan calls for retrieval before answering.

    TODO 2 contract:
    1. Inspect plan_steps (a list of short plan-step strings your planning
       prompt is instructed to produce).
    2. If any step contains the substring "search notes" (case-insensitive,
       ignoring leading/trailing whitespace on the step), return the string
       "retrieve".
    3. Otherwise, return the string "answer".
    """
    if any("search notes" in step.strip().lower() for step in plan_steps):
        return "retrieve"
    return "answer"


def extract_citations(answer_text: str, available_note_ids: list[str]) -> list[str]:
    """Pull valid note citations out of a draft answer.

    TODO 3 contract:
    1. Find every occurrence of the pattern "[NOTE-...]" in answer_text
       using CITATION_PATTERN (e.g. matches "[NOTE-3]", "[NOTE-legal-1]").
    2. Keep only the matches whose note ID is present in
       available_note_ids — a citation to a note that was never retrieved
       must be dropped, not kept.
    3. Remove duplicates, but preserve the order in which each valid note ID
       first appears in answer_text.
    4. Return the resulting list of note ID strings without brackets, e.g.
       ["NOTE-3", "NOTE-7"]. Return an empty list if there are none.
    """
    available_ids = set(available_note_ids)
    citations: list[str] = []
    seen: set[str] = set()
    for match in CITATION_PATTERN.finditer(answer_text):
        note_id = match.group(1)
        if note_id in available_ids and note_id not in seen:
            citations.append(note_id)
            seen.add(note_id)
    return citations


def reflect_and_maybe_revise(
    question: str,
    draft_answer: str,
    citations: list[str],
    available_note_ids: list[str],
    reflect_fn: Callable[[str, str, list[str]], str],
) -> str:
    """Check the draft's grounding and return the final answer text.

    TODO 4 contract:
    1. If available_note_ids is non-empty AND citations is empty (notes
       were retrieved but the draft cited none of them), the draft has
       failed grounding: call
       reflect_fn(question, draft_answer, available_note_ids) and return
       its result directly as the revised answer.
    2. Otherwise, call
       reflect_fn(question, draft_answer, available_note_ids) and inspect
       its return value:
       - If it is exactly the string "APPROVED", return draft_answer
         unchanged.
       - Otherwise, treat the returned string as a full replacement answer
         and return it instead of draft_answer.
    3. Always return a string; never return None.
    """
    reflection = reflect_fn(question, draft_answer, available_note_ids)
    if available_note_ids and not citations:
        return reflection if isinstance(reflection, str) else ""
    if reflection == "APPROVED":
        return draft_answer
    return reflection if isinstance(reflection, str) else ""


# ---------------------------------------------------------------------------
# Supplied: orchestration. Do not modify below this line.
# ---------------------------------------------------------------------------


def run_notebook_query(
    question: str,
    notebook_id: str,
    storage_dir: str,
    note_index: list[dict[str, Any]],
    plan_fn: Callable[[str], list[str]],
    answer_fn: Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], str],
    reflect_fn: Callable[[str, str, list[str]], str],
    summarize_fn: Callable[[list[dict[str, Any]]], str],
    keep_recent: int = 6,
    retrieval_k: int = 3,
) -> AgentState:
    """Supplied orchestrator. Calls your four functions in order. This is
    what both the tests and the app call — do not modify it."""
    state = resume_or_start_session(notebook_id, storage_dir, keep_recent, summarize_fn)
    state["messages"].append({"role": "user", "content": question})

    plan = plan_fn(question)
    state["plan"] = plan

    decision = route_after_plan(plan)
    retrieved = retrieve_top_k(note_index, question, retrieval_k) if decision == "retrieve" else []
    state["retrieved"] = retrieved

    draft = answer_fn(question, retrieved, state["messages"])
    available_note_ids = [item["note_id"] for item in retrieved]
    citations = extract_citations(draft, available_note_ids)

    final_answer = reflect_and_maybe_revise(question, draft, citations, available_note_ids, reflect_fn)
    final_citations = extract_citations(final_answer, available_note_ids)

    state["draft_answer"] = draft
    state["citations"] = final_citations
    state["final_answer"] = final_answer
    state["messages"].append({"role": "assistant", "content": final_answer})

    save_session(storage_dir, notebook_id, state)
    return state


def _load_prompt(name: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "prompts", name)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def build_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is required to call Groq.")
    return ChatGroq(model=os.getenv("GROQ_MODEL", DEFAULT_MODEL), api_key=api_key)


def make_plan_fn(llm: Any) -> Callable[[str], list[str]]:
    prompt = _load_prompt("plan_prompt.txt")

    def plan_fn(question: str) -> list[str]:
        response = llm.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": question}])
        lines = str(getattr(response, "content", "") or "").splitlines()
        return [line.strip("-* ").strip() for line in lines if line.strip()]

    return plan_fn


def make_answer_fn(llm: Any) -> Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], str]:
    prompt = _load_prompt("answer_prompt.txt")

    def answer_fn(question: str, retrieved: list[dict[str, Any]], messages: list[dict[str, Any]]) -> str:
        notes_block = "\n".join(f"[{item['note_id']}] {item['text']}" for item in retrieved)
        user_content = (
            f"Question: {question}\n\n"
            f"Retrieved notes:\n{notes_block if notes_block else '(none retrieved)'}"
        )
        response = llm.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": user_content}])
        return str(getattr(response, "content", "") or "")

    return answer_fn


def make_reflect_fn(llm: Any) -> Callable[[str, str, list[str]], str]:
    prompt = _load_prompt("reflect_prompt.txt")

    def reflect_fn(question: str, draft_answer: str, available_note_ids: list[str]) -> str:
        user_content = (
            f"Question: {question}\nAvailable note IDs: {available_note_ids}\nDraft answer:\n{draft_answer}"
        )
        response = llm.invoke([{"role": "system", "content": prompt}, {"role": "user", "content": user_content}])
        return str(getattr(response, "content", "") or "").strip()

    return reflect_fn


def _default_summarize_fn(older_messages: list[dict[str, Any]]) -> str:
    parts = [f"{m.get('role')}: {str(m.get('content'))[:80]}" for m in older_messages]
    return " | ".join(parts)


SAMPLE_NOTES = [
    {"note_id": "NOTE-1", "text": "LangGraph state fields need the add_messages reducer to merge instead of overwrite."},
    {"note_id": "NOTE-2", "text": "FastEmbed runs embeddings locally with no external API call needed."},
    {"note_id": "NOTE-3", "text": "A recursion limit on graph invoke guarantees termination even in a stuck loop."},
]


def main() -> None:
    llm = build_llm()
    note_index = build_index(SAMPLE_NOTES)
    result = run_notebook_query(
        question="Why do I need a reducer on my messages field in LangGraph?",
        notebook_id="demo-notebook",
        storage_dir=os.path.join(os.path.dirname(__file__), ".sessions"),
        note_index=note_index,
        plan_fn=make_plan_fn(llm),
        answer_fn=make_answer_fn(llm),
        reflect_fn=make_reflect_fn(llm),
        summarize_fn=_default_summarize_fn,
    )
    print(result["final_answer"])


if __name__ == "__main__":
    main()
