import tempfile
import unittest
from pathlib import Path
from typing import Any

from main import (
    SAMPLE_NOTES,
    _load_prompt,
    extract_citations,
    reflect_and_maybe_revise,
    resume_or_start_session,
    route_after_plan,
    run_notebook_query,
)
from utils.retrieval import build_index
from utils.storage import load_session, save_session


class SessionTests(unittest.TestCase):
    def test_new_notebook_starts_with_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as storage_dir:
            state = resume_or_start_session(
                "research-a", storage_dir, 4, lambda messages: "unused"
            )

        self.assertEqual(
            state,
            {
                "notebook_id": "research-a",
                "messages": [],
                "plan": [],
                "retrieved": [],
                "draft_answer": "",
                "citations": [],
                "final_answer": "",
            },
        )

    def test_short_history_is_restored_without_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as storage_dir:
            saved_state = {
                "notebook_id": "wrong-id",
                "messages": [{"role": "user", "content": "Earlier question"}],
                "plan": ["Search notes"],
                "retrieved": [],
                "draft_answer": "draft",
                "citations": [],
                "final_answer": "answer",
            }
            save_session(storage_dir, "research-a", saved_state)

            def unexpected_summarize(messages: list[dict[str, Any]]) -> str:
                self.fail("History at the limit must not be summarized")

            state = resume_or_start_session(
                "research-a", storage_dir, 1, unexpected_summarize
            )

        self.assertEqual(state["notebook_id"], "research-a")
        self.assertEqual(state["messages"], saved_state["messages"])
        self.assertEqual(state["final_answer"], "answer")

    def test_long_history_is_compacted_and_recent_messages_remain(self) -> None:
        with tempfile.TemporaryDirectory() as storage_dir:
            messages = [
                {"role": "user", "content": "Question one"},
                {"role": "assistant", "content": "Answer one"},
                {"role": "user", "content": "Question two"},
                {"role": "assistant", "content": "Answer two"},
            ]
            save_session(
                storage_dir,
                "research-a",
                {
                    "notebook_id": "research-a",
                    "messages": messages,
                    "plan": [],
                    "retrieved": [],
                    "draft_answer": "",
                    "citations": [],
                    "final_answer": "",
                },
            )
            summarized: list[list[dict[str, Any]]] = []

            def summarize(older: list[dict[str, Any]]) -> str:
                summarized.append(older)
                return "Earlier exchange summary"

            state = resume_or_start_session("research-a", storage_dir, 2, summarize)

        self.assertEqual(summarized, [messages[:2]])
        self.assertEqual(
            state["messages"],
            [
                {
                    "role": "system",
                    "content": "Summary of earlier conversation: Earlier exchange summary",
                },
                *messages[2:],
            ],
        )


class RoutingAndCitationTests(unittest.TestCase):
    def test_route_retrieves_for_case_insensitive_search_notes_substring(self) -> None:
        self.assertEqual(
            route_after_plan(["Identify relevant context", "  SEARCH NOTES for evidence  "]),
            "retrieve",
        )

    def test_route_answers_without_search_notes_step(self) -> None:
        self.assertEqual(route_after_plan(["Reply conversationally"]), "answer")
        self.assertEqual(route_after_plan([]), "answer")

    def test_citations_are_validated_deduplicated_and_ordered(self) -> None:
        answer = (
            "First [NOTE-legal-1], then [NOTE-unknown], "
            "next [NOTE-3], again [NOTE-legal-1]."
        )
        self.assertEqual(
            extract_citations(answer, ["NOTE-3", "NOTE-legal-1"]),
            ["NOTE-legal-1", "NOTE-3"],
        )

    def test_citations_without_available_notes_are_empty(self) -> None:
        self.assertEqual(extract_citations("Claim [NOTE-1].", []), [])
        self.assertEqual(extract_citations("No citations here.", ["NOTE-1"]), [])


class ReflectionTests(unittest.TestCase):
    def test_approved_draft_is_returned_unchanged(self) -> None:
        draft = "Grounded claim [NOTE-1]."
        result = reflect_and_maybe_revise(
            "Question?", draft, ["NOTE-1"], ["NOTE-1"],
            lambda question, text, ids: "APPROVED",
        )
        self.assertEqual(result, draft)

    def test_reflection_replacement_is_returned(self) -> None:
        result = reflect_and_maybe_revise(
            "Question?", "Draft [NOTE-1].", ["NOTE-1"], ["NOTE-1"],
            lambda question, text, ids: "Revised answer [NOTE-1].",
        )
        self.assertEqual(result, "Revised answer [NOTE-1].")

    def test_uncited_draft_with_retrieved_notes_is_revised(self) -> None:
        calls: list[tuple[str, str, list[str]]] = []

        def reflect(question: str, draft: str, note_ids: list[str]) -> str:
            calls.append((question, draft, note_ids))
            return "Corrected answer [NOTE-2]."

        result = reflect_and_maybe_revise(
            "Question?", "Uncited draft.", [], ["NOTE-2"], reflect
        )
        self.assertEqual(result, "Corrected answer [NOTE-2].")
        self.assertEqual(calls, [("Question?", "Uncited draft.", ["NOTE-2"])])

    def test_reflection_never_returns_none(self) -> None:
        result = reflect_and_maybe_revise(
            "Question?", "Draft", [], [], lambda question, text, ids: None
        )
        self.assertIsInstance(result, str)


class OrchestrationTests(unittest.TestCase):
    def test_general_question_skips_retrieval_and_history_is_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as storage_dir:
            note_index = build_index(SAMPLE_NOTES)
            observed_messages: list[list[dict[str, Any]]] = []

            def answer(question: str, retrieved: list[dict[str, Any]], messages: list[dict[str, Any]]) -> str:
                self.assertEqual(retrieved, [])
                observed_messages.append(list(messages))
                return "A direct response."

            def run(notebook_id: str) -> dict[str, Any]:
                return run_notebook_query(
                    question="Hello, how are you?",
                    notebook_id=notebook_id,
                    storage_dir=storage_dir,
                    note_index=note_index,
                    plan_fn=lambda question: ["Respond conversationally"],
                    answer_fn=answer,
                    reflect_fn=lambda question, draft, ids: "APPROVED",
                    summarize_fn=lambda messages: "summary",
                )

            first = run("notebook-a")
            second = run("notebook-a")
            separate = run("notebook-b")
            persisted = load_session(storage_dir, "notebook-a")

        self.assertEqual(first["retrieved"], [])
        self.assertEqual(len(first["messages"]), 2)
        self.assertEqual(len(second["messages"]), 4)
        self.assertEqual(len(separate["messages"]), 2)
        self.assertEqual(observed_messages[1][1]["content"], "A direct response.")
        self.assertEqual(len(observed_messages[2]), 1)
        self.assertEqual(len(persisted["messages"]), 4)

    def test_retrieved_notes_are_cited_and_uncited_draft_is_revised(self) -> None:
        with tempfile.TemporaryDirectory() as storage_dir:
            note_index = build_index(
                [{"note_id": "NOTE-1", "text": "LangGraph reducers merge message state updates."}]
            )
            result = run_notebook_query(
                question="How do LangGraph reducers merge message state updates?",
                notebook_id="research",
                storage_dir=storage_dir,
                note_index=note_index,
                plan_fn=lambda question: ["Search notes"],
                answer_fn=lambda question, retrieved, messages: "Uncited draft.",
                reflect_fn=lambda question, draft, ids: "Reducers merge updates [NOTE-1].",
                summarize_fn=lambda messages: "summary",
            )

        self.assertEqual([item["note_id"] for item in result["retrieved"]], ["NOTE-1"])
        self.assertEqual(result["final_answer"], "Reducers merge updates [NOTE-1].")
        self.assertEqual(result["citations"], ["NOTE-1"])
        self.assertEqual(result["messages"][-1]["content"], result["final_answer"])

    def test_prompts_exist_and_include_key_contracts(self) -> None:
        plan_prompt = _load_prompt("plan_prompt.txt")
        answer_prompt = _load_prompt("answer_prompt.txt")
        reflect_prompt = _load_prompt("reflect_prompt.txt")

        self.assertIn("Search notes", plan_prompt)
        self.assertIn("one action per line", plan_prompt)
        self.assertIn("[NOTE-3]", answer_prompt)
        self.assertIn("don't have a note", answer_prompt)
        self.assertIn("APPROVED", reflect_prompt)
        self.assertIn("available note IDs", reflect_prompt)


if __name__ == "__main__":
    unittest.main()
