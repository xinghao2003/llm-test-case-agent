# User Story Test Generator

Minimal Gradio application that uses an LLM-driven, agentic workflow to turn user stories or requirements into ready-to-use test plans and code for an uploaded codebase.

## Quick Start

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and populate `GOOGLE_API_KEY` plus any optional `GEMINI_*` tweaks.
3. Run `python app.py` and open the served URL.

## Current Capabilities

- **LLM**: Google Gemini (configurable via environment variables)
- **Primary language**: Python
- **Test framework**: pytest

## Examples

- `examples/user_story.txt` offers a sample requirement to quickly trial the User Story Test Generator workflow.
- `examples/login_app/` contains a lightweight web app you can upload to exercise end-to-end test generation.

## Roadmap

- Support additional hosted or self-managed LLM providers
- Extend language coverage beyond Python
- Add more testing frameworks and auxiliary tooling
- Evolve the agentic workflow with automated iteration, self-testing before delivery, improved coverage heuristics, and human-in-the-loop checkpoints to keep generated tests reliable out of the box
- Improve use of Gradio for better UI/UX
