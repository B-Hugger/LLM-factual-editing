"""
Drift and Edit Success Metrics

This module provides metrics for measuring:
1. Model drift: KL divergence between original and edited model distributions
2. Edit success: Changes in log-probability for target factual outputs
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class DriftMetrics:
    """
    Container for drift measurement results.

    Attributes:
        kl_divergences: Per-prompt KL divergence values
        mean_kl: Mean KL divergence across all prompts
        max_kl: Maximum KL divergence
        prompt_results: Detailed results per prompt
    """
    kl_divergences: List[float] = field(default_factory=list)
    mean_kl: float = 0.0
    max_kl: float = 0.0
    prompt_results: List[Dict] = field(default_factory=list)

    def __repr__(self):
        return (f"DriftMetrics(mean_kl={self.mean_kl:.6f}, "
                f"max_kl={self.max_kl:.6f}, "
                f"num_prompts={len(self.kl_divergences)})")


@dataclass
class EditSuccessMetrics:
    """
    Container for edit success measurement results.

    Attributes:
        target_log_prob_before: Log prob of target before edit
        target_log_prob_after: Log prob of target after edit
        log_prob_improvement: Improvement in log probability
        success: Whether edit achieved meaningful improvement
    """
    target_log_prob_before: float = 0.0
    target_log_prob_after: float = 0.0
    log_prob_improvement: float = 0.0
    rank_before: int = 0
    rank_after: int = 0
    success: bool = False

    def __repr__(self):
        return (f"EditSuccess(improvement={self.log_prob_improvement:.4f}, "
                f"rank: {self.rank_before} -> {self.rank_after}, "
                f"success={self.success})")


def compute_kl_divergence(
    logits_original: torch.Tensor,
    logits_edited: torch.Tensor,
    temperature: float = 1.0,
    reduce: str = "mean"
) -> torch.Tensor:
    """
    Compute KL divergence between original and edited model distributions.

    KL(P || Q) where P is original distribution and Q is edited distribution.

    Args:
        logits_original: Logits from original model [batch, vocab] or [batch, seq, vocab]
        logits_edited: Logits from edited model (same shape)
        temperature: Temperature for softmax (default 1.0)
        reduce: Reduction method ('mean', 'sum', 'none')

    Returns:
        KL divergence value(s)
    """
    # Apply temperature
    logits_original = logits_original / temperature
    logits_edited = logits_edited / temperature

    # Compute log probabilities
    log_probs_original = F.log_softmax(logits_original, dim=-1)
    log_probs_edited = F.log_softmax(logits_edited, dim=-1)

    # Compute probabilities for original (P)
    probs_original = F.softmax(logits_original, dim=-1)

    # KL(P || Q) = sum(P * (log P - log Q))
    kl = (probs_original * (log_probs_original - log_probs_edited)).sum(dim=-1)

    if reduce == "mean":
        return kl.mean()
    elif reduce == "sum":
        return kl.sum()
    else:
        return kl


def compute_last_token_kl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    baseline_logits: Dict[str, torch.Tensor],
    device: str = "cuda"
) -> DriftMetrics:
    """
    Compute KL divergence for the last token position across prompts.

    Args:
        model: The (potentially edited) model
        tokenizer: The tokenizer
        prompts: List of prompts to evaluate
        baseline_logits: Dict mapping prompts to original last-token logits
        device: Device to use

    Returns:
        DriftMetrics containing per-prompt and aggregate KL divergence
    """
    model.eval()
    kl_values = []
    prompt_results = []

    with torch.no_grad():
        for prompt in prompts:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

            # Get current logits
            outputs = model(**inputs)
            current_logits = outputs.logits[0, -1, :]  # Last token

            # Get baseline logits
            if prompt in baseline_logits:
                original_logits = baseline_logits[prompt].to(device)

                # Compute KL divergence
                kl = compute_kl_divergence(
                    original_logits.unsqueeze(0),
                    current_logits.unsqueeze(0),
                    reduce="mean"
                ).item()

                kl_values.append(kl)
                prompt_results.append({
                    "prompt": prompt,
                    "kl_divergence": kl,
                })
            else:
                print(f"Warning: No baseline for prompt '{prompt[:50]}...'")

    if kl_values:
        return DriftMetrics(
            kl_divergences=kl_values,
            mean_kl=sum(kl_values) / len(kl_values),
            max_kl=max(kl_values),
            prompt_results=prompt_results
        )
    else:
        return DriftMetrics()


def compute_edit_success(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    target: str,
    baseline_log_prob: Optional[float] = None,
    device: str = "cuda",
    success_threshold: float = 2.0  # Log prob improvement threshold
) -> EditSuccessMetrics:
    """
    Measure edit success based on target log probability changes.

    Args:
        model: The (potentially edited) model
        tokenizer: The tokenizer
        prompt: The input prompt (e.g., "The Eiffel Tower is located in")
        target: The target completion (e.g., " Rome")
        baseline_log_prob: Log prob from original model (computed if not provided)
        device: Device to use
        success_threshold: Minimum log prob improvement to consider success

    Returns:
        EditSuccessMetrics with detailed results
    """
    model.eval()

    # Ensure target starts with space if needed
    if not target.startswith(" "):
        target = " " + target

    # Tokenize prompt and target
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    target_ids = tokenizer.encode(target, add_special_tokens=False)

    with torch.no_grad():
        # Get model output for prompt
        outputs = model(prompt_ids)
        logits = outputs.logits[0, -1, :]  # Last token position

        # Get log probability of first target token
        log_probs = F.log_softmax(logits, dim=-1)
        target_first_token = target_ids[0]
        current_log_prob = log_probs[target_first_token].item()

        # Get rank of target token
        sorted_indices = torch.argsort(log_probs, descending=True)
        rank = (sorted_indices == target_first_token).nonzero().item() + 1

    # Calculate improvement
    if baseline_log_prob is not None:
        improvement = current_log_prob - baseline_log_prob
    else:
        improvement = 0.0

    return EditSuccessMetrics(
        target_log_prob_before=baseline_log_prob if baseline_log_prob else 0.0,
        target_log_prob_after=current_log_prob,
        log_prob_improvement=improvement,
        rank_before=0,  # Would need baseline model to compute
        rank_after=rank,
        success=improvement >= success_threshold or rank <= 10
    )


def batch_compute_edit_success(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    edit_requests: List[Dict],
    baseline_log_probs: Optional[Dict[str, float]] = None,
    device: str = "cuda"
) -> List[EditSuccessMetrics]:
    """
    Compute edit success metrics for multiple edit requests.

    Args:
        model: The edited model
        tokenizer: The tokenizer
        edit_requests: List of dicts with 'prompt', 'subject', 'target_new'
        baseline_log_probs: Optional dict mapping prompts to baseline log probs
        device: Device to use

    Returns:
        List of EditSuccessMetrics for each request
    """
    results = []
    for request in edit_requests:
        prompt = request["prompt"].format(request["subject"])
        target = request["target_new"]["str"]

        baseline = None
        if baseline_log_probs and prompt in baseline_log_probs:
            baseline = baseline_log_probs[prompt]

        metrics = compute_edit_success(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target=target,
            baseline_log_prob=baseline,
            device=device
        )
        results.append(metrics)

    return results
