# Nura

**Your data. Your model. Gets smarter every day.**

Connect your data. Train with RL. Ship an API that improves itself automatically.

---

Nura is an open-source Python SDK that lets any company connect their data, train a custom AI model using reinforcement learning, and get a managed API that improves automatically from real-world outcomes — no ML team required.

[![CI](https://github.com/ThilakNarasimhamurthy/nura/actions/workflows/ci.yml/badge.svg)](https://github.com/ThilakNarasimhamurthy/nura/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

---

## Real results

Tested on the [Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (26,872 real interactions):

| | Score |
|---|---|
| Before training | 31% |
| After 20 GRPO steps | **33% (+6.5%)** |

Model: `Qwen/Qwen2.5-0.5B-Instruct` · Hardware: Apple Silicon (MPS) · Steps: 20

> 20 steps is a short run. 100+ steps on GPU compounds this significantly.

---

## Why Nura?

Most AI fine-tuning tools are built for researchers. Nura is built for businesses.

| Problem | Nura's answer |
|---|---|
| Fine-tuning requires a data science team | 3-line Python API anyone can follow |
| Models go stale after training | Continuous RL loop improves from real outcomes |
| Black-box results nobody understands | Plain-English explanations via the Brain |
| Vendor lock-in | Swap adapters: Ollama → OpenAI → any model |

---

## Quick start

```bash
pip install nura-sdk          # coming soon — for now, clone and install locally
ollama pull llama3.2:1b       # pull a local model
```

```python
import asyncio
from sdk.nura import OllamaAdapter, ResolutionReward, OllamaBrain

async def main():
    # 1. Connect a model
    adapter = OllamaAdapter(model="llama3.2:1b")

    # 2. Define success (1.0 = resolved, 0.0 = escalated)
    reward = ResolutionReward()

    # 3. Ask the Brain to explain your data
    brain = OllamaBrain()
    summary = await brain.analyze_dataset([
        {"message": "How do I reset my password?", "resolved": True},
        {"message": "I was charged twice", "resolved": False},
    ])
    print(summary)

    # 4. Generate a response
    reply = await adapter.generate(
        prompt="How do I reset my password?",
        context="You are a helpful customer support agent.",
    )
    print(reply)

asyncio.run(main())
```

---

## Architecture

```
nura/
├── sdk/nura/
│   ├── adapters/        # LLM backends (Ollama, OpenAI, …)
│   ├── rewards/         # Reward functions (Resolution, CSAT, …)
│   ├── brain/           # Plain-English explainer layer
│   ├── training/        # RL training loop (GRPO)
│   ├── data/            # Data connectors
│   └── registry/        # Model registry
├── api/                 # FastAPI backend
├── dashboard/           # Next.js frontend
├── examples/            # End-to-end use-case scripts
└── tests/
```

---

## Core concepts

### Adapters
Wrap any LLM so the rest of the SDK stays model-agnostic.

```python
from sdk.nura import OllamaAdapter
adapter = OllamaAdapter(model="llama3.2:1b")
response = await adapter.generate("Summarise this ticket: ...")
```

### Rewards
Map real business outcomes to a training signal.

```python
from sdk.nura import ResolutionReward
reward = ResolutionReward()
scores = reward.score(prompts, completions, outcomes=[1.0, 0.0, 1.0])
```

### Brain
Explains what's happening in plain English — no jargon.

```python
from sdk.nura import OllamaBrain
brain = OllamaBrain()
explanation = await brain.explain_result(before_score=0.61, after_score=0.74, metrics={})
# → "Your AI now correctly handles 13 more conversations out of every 100..."
```

---

## Roadmap

- [x] Core SDK: adapters, rewards, brain
- [x] GRPO training loop (LoRA fine-tuning via TRL)
- [x] Data connectors (CSV, JSONL, HuggingFace datasets)
- [x] `nura train` CLI
- [ ] More data connectors (Intercom, Zendesk, SQL)
- [ ] Model registry
- [ ] FastAPI serving layer
- [ ] Dashboard (Next.js)
- [ ] `pip install nura-sdk` release
- [ ] OpenAI + Anthropic adapters
- [ ] Hosted cloud version

---

## Contributing

We welcome contributions of all kinds — new adapters, new reward functions, vertical examples, and bug fixes. See [CONTRIBUTING.md](CONTRIBUTING.md) to get started in 3 commands.

---

## License

MIT — free to use, modify, and distribute. See [LICENSE](LICENSE).
