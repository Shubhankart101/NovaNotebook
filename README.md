# Nova Notebook

Nova Notebook is a personal research assistant that plans each question, searches a notebook's local notes when needed, cites retrieved evidence, checks its draft, and stores the conversation per notebook so it can be resumed later.

## Requirements

- Python 3.10 or newer
- A Groq API key to run the live LLM demo
- Internet access for the live Groq calls; note retrieval itself runs locally

The unit tests use deterministic callbacks and do not call an LLM or require an API key.

## Setup on Windows

From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set the key for the current PowerShell session and run the demo:

```powershell
$env:GROQ_API_KEY = "your-groq-api-key"
python main.py
```

`GROQ_MODEL` is optional. When unset, the app uses `openai/gpt-oss-120b`. The demo uses the sample notes in `main.py` and saves its history under `.sessions/`.

## Run the Tests

```powershell
python -m unittest discover -s tests -v
```

The suite is local and deterministic. It does not read or write the app's `.sessions/` directory; session tests use temporary directories.

## How the Pipeline Works

1. `resume_or_start_session` loads the state for the requested notebook, creates the correctly shaped empty state for a new notebook, and compacts old messages when history exceeds the configured limit.
2. The planning prompt emits short ordered steps. If a step contains `Search notes`, the router retrieves relevant notes; otherwise the question goes straight to drafting without retrieval.
3. The answer prompt drafts a concise response and cites note-dependent claims using the exact retrieved IDs, such as `[NOTE-1]`.
4. Citation extraction keeps only IDs that were actually retrieved, removes duplicates, and preserves their first-appearance order.
5. The reflection prompt checks grounding. When notes were retrieved but the draft cites none, the pipeline requires a replacement response before returning an answer.
6. The final question and answer are saved in that notebook's JSON session file. A later call with the same notebook ID resumes that history; a different ID uses a separate file and conversation.

The local retriever expects a prebuilt index. Build one from records shaped like `{"note_id": "NOTE-1", "text": "..."}` with `utils.retrieval.build_index(notes)`. The command-line demo does this for `SAMPLE_NOTES` before it calls the orchestrator.

## What the Tests Cover

- A never-used notebook returns empty messages, plan, retrieved notes, citations, and answers.
- Short history is restored without compaction; long history is summarized while recent messages remain.
- Notebook IDs are kept distinct, and saved history is visible on the next query for the same notebook.
- Planning routes case-insensitive `Search notes` steps to retrieval and leaves general questions on the direct-answer path.
- Citation IDs are filtered to retrieved notes, deduplicated, and returned in appearance order.
- Reflection preserves an approved draft, accepts a replacement, and revises an uncited draft when notes are available.
- The orchestration persists final answers, returns citations from retrieved notes, and keeps prompts present with their key instructions.

These checks map to the nine acceptance criteria in the assignment screenshots. LLM wording and model judgment are not deterministic unit-test targets; the tests instead verify the pipeline contracts with fixed callbacks.

## Current Scope

The required pipeline performs one retrieval for the full question after planning. Planning can produce multiple ordered steps, but separate retrieval and synthesis for each sub-question is an optional extension and is not implemented. Notes are supplied as an in-memory index in the demo; persistence here covers conversation state, not a user-editable note database.
