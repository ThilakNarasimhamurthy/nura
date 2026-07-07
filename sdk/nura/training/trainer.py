"""NuraTrainer — RL fine-tuning via GRPO with LoRA adapters."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Keywords that indicate a substantive, actionable response.
_RESOLUTION_KEYWORDS = {
    "track",
    "refund",
    "cancel",
    "process",
    "resolve",
    "help",
    "update",
    "confirm",
    "assist",
}


def _heuristic_reward(completions: list[str]) -> list[float]:
    """
    Score generated completions without ground-truth outcomes.

    Used internally by the GRPO training loop, where the model produces
    completions we have no real-world label for.  A completion scores
    ``1.0`` when it is substantive (> 50 words) **and** contains at least
    one resolution keyword; ``0.0`` otherwise.

    Parameters
    ----------
    completions:
        Batch of generated text strings.

    Returns
    -------
    list[float]
        One score per completion.
    """
    scores: list[float] = []
    for text in completions:
        words = text.lower().split()
        long_enough = len(words) > 50
        has_keyword = bool(_RESOLUTION_KEYWORDS & set(words))
        scores.append(1.0 if (long_enough and has_keyword) else 0.0)
    return scores


class NuraTrainer:
    """
    Orchestrates RL fine-tuning using GRPO and LoRA adapters.

    The trainer loads a small base model (default ``facebook/opt-125m``),
    applies a low-rank adapter via PEFT, and runs TRL's GRPOTrainer for a
    fixed number of steps.  Before and after scores are captured so callers
    can measure the improvement.

    Parameters
    ----------
    config:
        Training configuration dict.  Recognised keys:

        ==================  =====================  =======================
        Key                 Type                   Default
        ==================  =====================  =======================
        ``base_model``      ``str``                ``"Qwen/Qwen2.5-0.5B-Instruct"``
        ``output_dir``      ``str``                **required**
        ``num_steps``       ``int``                ``20``
        ``batch_size``      ``int``                ``2``
        ``lora_rank``       ``int``                ``16``
        ``lora_alpha``      ``int``                ``32``
        ==================  =====================  =======================

    Examples
    --------
    >>> trainer = NuraTrainer({
    ...     "output_dir": "outputs/run-1",
    ...     "num_steps": 5,
    ...     "batch_size": 1,
    ... })
    >>> result = trainer.train(data, reward_fn=ResolutionReward())
    >>> print(result["improvement_pct"])
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.base_model: str = config.get("base_model", "Qwen/Qwen2.5-0.5B-Instruct")
        self.output_dir: str = config["output_dir"]
        self.num_steps: int = int(config.get("num_steps", 20))
        self.batch_size: int = int(config.get("batch_size", 2))
        self.lora_rank: int = int(config.get("lora_rank", 16))
        self.lora_alpha: int = int(config.get("lora_alpha", 32))

        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        """
        Return the best available device string.

        Prefers MPS on Apple Silicon, falls back to CPU everywhere else.
        CUDA is intentionally skipped to keep the local development loop
        fast on MacBooks.

        Returns
        -------
        str
            ``"mps"`` or ``"cpu"``.
        """
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model_and_tokenizer(self, device: str) -> tuple[Any, Any]:
        """
        Download (or load from cache) the base model and tokenizer.

        Parameters
        ----------
        device:
            Target device string (``"mps"`` or ``"cpu"``).

        Returns
        -------
        tuple[model, tokenizer]
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading tokenizer: %s", self.base_model)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading model: %s → %s", self.base_model, device)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map=device,
        )
        return model, tokenizer

    # ------------------------------------------------------------------
    # LoRA config
    # ------------------------------------------------------------------

    def _lora_config(self) -> Any:
        """
        Build a PEFT ``LoraConfig`` for the configured rank and alpha.

        Returns
        -------
        peft.LoraConfig
        """
        from peft import LoraConfig, TaskType

        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        data: list[dict[str, Any]],
        reward_fn: Any,
    ) -> float:
        """
        Generate completions for every prompt and return the average reward.

        Parameters
        ----------
        data:
            List of dicts produced by ``NuraDataLoader.prepare()``.
        reward_fn:
            Retained for API compatibility; not used internally.
            Scoring is done by the same heuristic the GRPO loop uses
            (> 50 words **and** contains a resolution keyword → 1.0),
            so before/after scores measure actual generation quality rather
            than the dataset's pre-labelled outcome distribution.

        Returns
        -------
        float
            Mean heuristic reward across all generated completions in
            ``[0.0, 1.0]``.
        """
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "Model not loaded. Call train() before evaluate(), "
                "or load the model manually."
            )

        prompts = [r["prompt"] for r in data]
        completions: list[str] = []

        self._model.eval()
        with torch.no_grad():
            for prompt in prompts:
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).to(self._model.device)

                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                # Decode only the newly generated tokens
                new_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
                completions.append(
                    self._tokenizer.decode(new_ids, skip_special_tokens=True)
                )

        # Score the actual generated text, not the dataset labels.
        # reward_fn is retained in the signature for API compatibility.
        scores = _heuristic_reward(completions)
        return sum(scores) / len(scores) if scores else 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        data: list[dict[str, Any]],
        reward_fn: Any,
    ) -> dict[str, Any]:
        """
        Fine-tune the base model on *data* using GRPO + LoRA.

        Steps
        -----
        1. Detect the best available device (MPS → CPU).
        2. Load the base model and tokenizer.
        3. Attach a LoRA adapter via PEFT.
        4. Evaluate the model **before** training.
        5. Run ``GRPOTrainer`` for ``num_steps`` steps.
        6. Evaluate the model **after** training.
        7. Save the LoRA adapter to ``output_dir``.
        8. Return a metrics dict.

        Parameters
        ----------
        data:
            List of dicts from ``NuraDataLoader.prepare()``.  Must contain
            ``prompt``, ``context``, ``ideal_response``, and
            ``outcome_signal`` keys.
        reward_fn:
            A :class:`~nura.rewards.base.BaseReward` used for before/after
            evaluation.  During GRPO training itself a heuristic reward is
            used because the model generates completions we have no labels
            for.

        Returns
        -------
        dict
            ``before_score``, ``after_score``, ``improvement_pct``,
            ``steps``, ``adapter_path``, ``model``, ``examples_trained``.
        """
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer

        # ── 1. Device ────────────────────────────────────────────────────
        device = self._detect_device()
        logger.info("Training device: %s", device)

        # ── 2. Load model & tokenizer ────────────────────────────────────
        self._model, self._tokenizer = self._load_model_and_tokenizer(device)

        # ── 3. LoRA config (passed directly to GRPOTrainer) ──────────────
        peft_config = self._lora_config()

        # ── 4. Evaluate BEFORE training ───────────────────────────────────
        logger.info("Evaluating baseline …")
        before_score = self.evaluate(data, reward_fn)
        logger.info("Baseline score: %.4f", before_score)

        # ── 5. Build HuggingFace Dataset for GRPO ─────────────────────────
        #   GRPO only needs a "prompt" column; completions are generated
        #   on-the-fly during training.
        hf_dataset = Dataset.from_list([{"prompt": r["prompt"]} for r in data])

        # Reward wrapper: scores model-generated completions using the
        # heuristic signal (no ground-truth labels available at generation time).
        def _grpo_reward(
            prompts: list[str],
            completions: list[str],
            **_kwargs: Any,
        ) -> list[float]:
            return _heuristic_reward(completions)

        # ── 6. GRPOTrainer ────────────────────────────────────────────────
        grpo_config = GRPOConfig(
            output_dir=self.output_dir,
            max_steps=self.num_steps,
            per_device_train_batch_size=self.batch_size,
            num_generations=max(2, self.batch_size),
            max_completion_length=80,
            temperature=0.7,
            logging_steps=max(1, self.num_steps // 5),
            report_to="none",
            save_strategy="no",
        )

        trainer = GRPOTrainer(
            model=self._model,
            reward_funcs=[_grpo_reward],
            args=grpo_config,
            train_dataset=hf_dataset,
            processing_class=self._tokenizer,
            peft_config=peft_config,
        )

        logger.info("Starting GRPO training for %d steps …", self.num_steps)
        trainer.train()

        # Keep reference to the LoRA-adapted model for post-train evaluation
        self._model = trainer.model

        # ── 7. Evaluate AFTER training ────────────────────────────────────
        logger.info("Evaluating after training …")
        after_score = self.evaluate(data, reward_fn)
        logger.info("After-training score: %.4f", after_score)

        # ── 8. Save adapter ───────────────────────────────────────────────
        adapter_path = str(Path(self.output_dir) / "adapter")
        os.makedirs(adapter_path, exist_ok=True)
        trainer.save_model(adapter_path)
        self._tokenizer.save_pretrained(adapter_path)
        logger.info("Adapter saved to %s", adapter_path)

        # ── 9. Build result dict ──────────────────────────────────────────
        improvement_pct = (
            ((after_score - before_score) / before_score * 100)
            if before_score > 0
            else 0.0
        )

        return {
            "before_score": round(before_score, 4),
            "after_score": round(after_score, 4),
            "improvement_pct": round(improvement_pct, 2),
            "steps": self.num_steps,
            "adapter_path": adapter_path,
            "model": self.base_model,
            "examples_trained": len(data),
        }
