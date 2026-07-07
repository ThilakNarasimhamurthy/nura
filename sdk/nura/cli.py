"""Nura command-line interface."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from .data.loader import NuraDataLoader
from .rewards.resolution import ResolutionReward
from .training.trainer import NuraTrainer

app = typer.Typer(
    name="nura",
    help="Your data. Your model. Gets smarter every day.",
    add_completion=False,
)
console = Console()


@app.command("version")
def version() -> None:
    """Show the Nura SDK version."""
    from . import __version__

    console.print(f"nura {__version__}")


@app.command("train")
def train(
    data: str = typer.Option(..., "--data", help="'bitext', a .jsonl, or a .csv path"),
    steps: int = typer.Option(20, "--steps", help="Number of GRPO training steps"),
    output: str = typer.Option(..., "--output", help="Directory to save the adapter"),
    model: str = typer.Option(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--model",
        help="Base HuggingFace model to fine-tune",
    ),
    n: int = typer.Option(200, "--n", help="Number of training examples to prepare"),
    batch_size: int = typer.Option(2, "--batch-size", help="Per-device batch size"),
) -> None:
    """Load data, run GRPO fine-tuning, and save a LoRA adapter."""
    console.print(
        Panel.fit(
            f"[bold]Nura Training Run[/bold]\n"
            f"data={data}  steps={steps}  model={model}\n"
            f"output={output}",
            border_style="cyan",
        )
    )

    # ── Load & prepare data ───────────────────────────────────────────
    console.print("[cyan]Loading dataset…[/cyan]")
    loader = NuraDataLoader(data)
    prepared = loader.prepare(n)
    loader.save(prepared, f"{output}/train.jsonl")

    console.print(f"[green]✓[/green] {loader.summary(prepared)}")
    console.print(
        f"[green]✓[/green] Baseline score: {loader.baseline_score(prepared):.2%}"
    )

    # ── Train ─────────────────────────────────────────────────────────
    console.print("\n[cyan]Starting training…[/cyan]")
    trainer = NuraTrainer(
        {
            "base_model": model,
            "output_dir": output,
            "num_steps": steps,
            "batch_size": batch_size,
        }
    )
    metrics = trainer.train(prepared, ResolutionReward())

    # ── Report ────────────────────────────────────────────────────────
    direction = "▲" if metrics["after_score"] >= metrics["before_score"] else "▼"
    console.print(
        Panel.fit(
            f"[bold green]Training complete[/bold green]\n\n"
            f"  Before : {metrics['before_score']:.2%}\n"
            f"  After  : {metrics['after_score']:.2%}  {direction}\n"
            f"  Change : {metrics['improvement_pct']:+.1f}%\n"
            f"  Steps  : {metrics['steps']}\n"
            f"  Adapter: {metrics['adapter_path']}",
            border_style="green",
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
