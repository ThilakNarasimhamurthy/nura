# Contributing to Nura

First off — thank you for considering contributing. Nura is built by people who believe
business owners shouldn't need a PhD to use AI. Every improvement helps.

---

## Set up your dev environment (3 commands)

```bash
git clone https://github.com/ThilakNarasimhamurthy/nura.git && cd nura
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e "sdk/[dev]" torch transformers trl peft fastapi uvicorn sqlalchemy \
    typer rich pydantic ollama pytest black ruff
```

> **Note:** You'll also need [Ollama](https://ollama.com) running locally with at least one model pulled:
> ```bash
> ollama pull llama3.2:1b
> ```

---

## Run the tests

```bash
pytest tests/
```

For formatting and linting:

```bash
black --check sdk/ tests/         # check formatting
ruff check sdk/ tests/            # check style
black sdk/ tests/                 # auto-fix formatting
```

---

## How to open a pull request

1. **Fork** the repo and create a branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Write your code and tests.
3. Make sure `black --check` and `pytest tests/` both pass.
4. Open a PR against `main`. Describe *what* it does and *why* — a sentence or two is enough.
5. A maintainer will review within a few days.

---

## What kinds of contributions are most welcome

### New reward functions
Got a reward that worked well for your use-case? Add it to `sdk/nura/rewards/`.
Subclass `BaseReward`, implement `score()` and `validate()`, add a test, and share it.

Examples we'd love to see:
- `SatisfactionReward` — normalised CSAT score
- `LatencyReward` — penalises verbose responses
- `SafetyReward` — integrates a toxicity classifier

### New adapters
Want to use OpenAI, Anthropic, or a HuggingFace model instead of Ollama?
Add an adapter to `sdk/nura/adapters/` that subclasses `BaseLLMAdapter`.

### New vertical examples
The `examples/` folder is where real use-cases live. Customer support is first —
e-commerce, HR, legal, and healthcare are next. If you have data and a domain,
open a PR with an example end-to-end script.

### Bug fixes
If something is broken, please open an issue first so we can discuss the fix together.
Small bugs (typos, off-by-one errors, wrong defaults) can go straight to a PR.

---

## Code style

- **Python 3.12+**, type hints everywhere, docstrings on every public class and method.
- `black` for formatting, `ruff` for linting — both run in CI.
- Keep functions small. If a function needs more than ~30 lines, consider splitting it.
- No `print()` in library code — use `logging` or `rich.console`.

---

## Questions?

Open a [GitHub Discussion](https://github.com/ThilakNarasimhamurthy/nura/discussions)
or file an issue. We're friendly, we promise.
