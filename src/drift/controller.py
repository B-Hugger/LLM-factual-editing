"""
Drift Controller

Implements controlled update application with scaling factor optimization.
Uses binary search to find optimal alpha that maintains edit success while
keeping drift below a threshold.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import (
    DriftMetrics,
    EditSuccessMetrics,
    compute_last_token_kl,
    compute_edit_success
)
from .baseline import BaselinePromptsManager
from ..editing.weight_update import WeightUpdate, apply_weight_update_context


@dataclass
class OptimizationResult:
    """
    Result of scaling factor optimization.

    Attributes:
        optimal_alpha: The optimal scaling factor found
        final_drift: Drift metrics at optimal alpha
        final_edit_success: Edit success metrics at optimal alpha
        search_history: History of alpha values and metrics explored
        converged: Whether the search converged within tolerance
    """
    optimal_alpha: float = 1.0
    final_drift: Optional[DriftMetrics] = None
    final_edit_success: Optional[EditSuccessMetrics] = None
    search_history: List[Dict] = field(default_factory=list)
    converged: bool = False

    def __repr__(self):
        return (f"OptimizationResult(alpha={self.optimal_alpha:.4f}, "
                f"drift={self.final_drift.mean_kl if self.final_drift else 'N/A':.6f}, "
                f"converged={self.converged})")


class DriftController:
    """
    Controller for drift-aware model editing.

    Implements binary search to find optimal scaling factor that:
    1. Maintains required edit success (factual improvement)
    2. Keeps drift on unrelated prompts below threshold
    """

    def __init__(
        self,
        baseline_manager: BaselinePromptsManager,
        drift_threshold: float = 0.1,
        edit_success_threshold: float = 1.0,
        alpha_tolerance: float = 0.01,
        max_iterations: int = 20,
        device: str = "cuda"
    ):
        """
        Initialize the drift controller.

        Args:
            baseline_manager: Manager for baseline prompts and logits
            drift_threshold: Maximum allowed mean KL divergence
            edit_success_threshold: Minimum required log prob improvement
            alpha_tolerance: Convergence tolerance for alpha
            max_iterations: Maximum binary search iterations
            device: Device to use for computation
        """
        self.baseline_manager = baseline_manager
        self.drift_threshold = drift_threshold
        self.edit_success_threshold = edit_success_threshold
        self.alpha_tolerance = alpha_tolerance
        self.max_iterations = max_iterations
        self.device = device

    def evaluate_at_alpha(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        weight_update: WeightUpdate,
        edit_request: Dict,
        alpha: float
    ) -> Tuple[DriftMetrics, EditSuccessMetrics]:
        """
        Evaluate drift and edit success at a specific scaling factor.

        Args:
            model: The model (will be temporarily modified)
            tokenizer: The tokenizer
            weight_update: The weight update to apply
            edit_request: The edit request for success measurement
            alpha: The scaling factor

        Returns:
            Tuple of (DriftMetrics, EditSuccessMetrics)
        """
        with apply_weight_update_context(model, weight_update, alpha=alpha):
            # Compute drift
            drift = compute_last_token_kl(
                model=model,
                tokenizer=tokenizer,
                prompts=self.baseline_manager.prompts,
                baseline_logits=self.baseline_manager.baseline_logits,
                device=self.device
            )

            # Compute edit success
            prompt = edit_request["prompt"].format(edit_request["subject"])
            target = edit_request["target_new"]["str"]

            # Get baseline log prob for target (from unedited model)
            # We need to compute this outside the context manager
            baseline_log_prob = None

            edit_success = compute_edit_success(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target=target,
                baseline_log_prob=baseline_log_prob,
                device=self.device
            )

        return drift, edit_success

    def find_optimal_alpha(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        weight_update: WeightUpdate,
        edit_request: Dict,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
    ) -> OptimizationResult:
        """
        Find optimal scaling factor using binary search.

        The search finds the largest alpha where:
        - drift < drift_threshold
        - edit_success >= edit_success_threshold (if possible)

        Args:
            model: The model to edit
            tokenizer: The tokenizer
            weight_update: The weight update to apply
            edit_request: The edit request
            alpha_min: Minimum alpha to consider
            alpha_max: Maximum alpha to consider

        Returns:
            OptimizationResult with optimal alpha and metrics
        """
        history = []

        # First, compute baseline edit success (at alpha=0)
        with apply_weight_update_context(model, weight_update, alpha=0.0):
            prompt = edit_request["prompt"].format(edit_request["subject"])
            target = edit_request["target_new"]["str"]
            baseline_edit = compute_edit_success(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target=target,
                device=self.device
            )
        baseline_log_prob = baseline_edit.target_log_prob_after

        print(f"Baseline target log prob: {baseline_log_prob:.4f}")
        print(f"Starting binary search for optimal alpha...")
        print(f"  Drift threshold: {self.drift_threshold}")
        print(f"  Edit success threshold: {self.edit_success_threshold}")

        best_alpha = alpha_min
        best_drift = None
        best_edit_success = None

        for iteration in range(self.max_iterations):
            alpha = (alpha_min + alpha_max) / 2

            # Evaluate at current alpha
            drift, edit_success = self.evaluate_at_alpha(
                model, tokenizer, weight_update, edit_request, alpha
            )

            # Update edit success with baseline comparison
            edit_success.target_log_prob_before = baseline_log_prob
            edit_success.log_prob_improvement = (
                edit_success.target_log_prob_after - baseline_log_prob
            )

            history.append({
                "iteration": iteration,
                "alpha": alpha,
                "mean_kl": drift.mean_kl,
                "max_kl": drift.max_kl,
                "log_prob_improvement": edit_success.log_prob_improvement,
                "target_rank": edit_success.rank_after
            })

            print(f"  Iter {iteration}: alpha={alpha:.4f}, "
                  f"drift={drift.mean_kl:.6f}, "
                  f"improvement={edit_success.log_prob_improvement:.4f}")

            # Check if drift is acceptable
            if drift.mean_kl <= self.drift_threshold:
                # Drift is acceptable, try larger alpha
                if alpha > best_alpha:
                    best_alpha = alpha
                    best_drift = drift
                    best_edit_success = edit_success
                alpha_min = alpha
            else:
                # Drift too high, try smaller alpha
                alpha_max = alpha

            # Check convergence
            if alpha_max - alpha_min < self.alpha_tolerance:
                print(f"  Converged at alpha={best_alpha:.4f}")
                return OptimizationResult(
                    optimal_alpha=best_alpha,
                    final_drift=best_drift,
                    final_edit_success=best_edit_success,
                    search_history=history,
                    converged=True
                )

        # Didn't converge, return best found
        print(f"  Max iterations reached, best alpha={best_alpha:.4f}")
        return OptimizationResult(
            optimal_alpha=best_alpha,
            final_drift=best_drift,
            final_edit_success=best_edit_success,
            search_history=history,
            converged=False
        )

    def apply_stabilized_edit(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        weight_update: WeightUpdate,
        edit_request: Dict,
        use_binary_search: bool = True,
        fixed_alpha: Optional[float] = None
    ) -> Tuple[AutoModelForCausalLM, OptimizationResult]:
        """
        Apply a stabilized edit to the model.

        Args:
            model: The model to edit
            tokenizer: The tokenizer
            weight_update: The weight update to apply
            edit_request: The edit request
            use_binary_search: Whether to use binary search for alpha
            fixed_alpha: If provided, use this fixed alpha instead of search

        Returns:
            Tuple of (modified model, optimization result)
        """
        if fixed_alpha is not None:
            # Use fixed alpha
            alpha = fixed_alpha
            drift, edit_success = self.evaluate_at_alpha(
                model, tokenizer, weight_update, edit_request, alpha
            )
            result = OptimizationResult(
                optimal_alpha=alpha,
                final_drift=drift,
                final_edit_success=edit_success,
                converged=True
            )
        elif use_binary_search:
            # Find optimal alpha
            result = self.find_optimal_alpha(
                model, tokenizer, weight_update, edit_request
            )
            alpha = result.optimal_alpha
        else:
            alpha = 1.0
            drift, edit_success = self.evaluate_at_alpha(
                model, tokenizer, weight_update, edit_request, alpha
            )
            result = OptimizationResult(
                optimal_alpha=alpha,
                final_drift=drift,
                final_edit_success=edit_success,
                converged=True
            )

        # Permanently apply the scaled update
        from ..editing.weight_update import apply_weight_update
        model = apply_weight_update(model, weight_update, alpha=alpha)

        print(f"Applied stabilized edit with alpha={alpha:.4f}")
        return model, result


def evaluate_sequential_edits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    weight_updates: List[WeightUpdate],
    edit_requests: List[Dict],
    baseline_manager: BaselinePromptsManager,
    controller: DriftController,
    use_stabilization: bool = True
) -> Dict:
    """
    Evaluate sequential editing: apply multiple edits one after another.

    Tracks:
    - Retention of earlier edits
    - Accumulation of drift
    - Performance degradation

    Args:
        model: The model to edit
        tokenizer: The tokenizer
        weight_updates: List of weight updates to apply sequentially
        edit_requests: Corresponding edit requests
        baseline_manager: Manager for baseline prompts
        controller: Drift controller
        use_stabilization: Whether to use drift-controlled application

    Returns:
        Dictionary with detailed results
    """
    results = {
        "num_edits": len(weight_updates),
        "per_edit_results": [],
        "cumulative_drift": [],
        "edit_retention": [],  # Track if earlier edits are retained
        "method": "stabilized" if use_stabilization else "baseline"
    }

    for i, (update, request) in enumerate(zip(weight_updates, edit_requests)):
        print(f"\n=== Edit {i+1}/{len(weight_updates)} ===")
        print(f"Request: {request['prompt'].format(request['subject'])} -> {request['target_new']['str']}")

        if use_stabilization:
            model, opt_result = controller.apply_stabilized_edit(
                model, tokenizer, update, request
            )
            edit_result = {
                "edit_index": i,
                "alpha": opt_result.optimal_alpha,
                "drift": opt_result.final_drift.mean_kl if opt_result.final_drift else None,
                "edit_success": opt_result.final_edit_success.log_prob_improvement if opt_result.final_edit_success else None
            }
        else:
            # Apply full update without stabilization
            from ..editing.weight_update import apply_weight_update
            model = apply_weight_update(model, update, alpha=1.0)

            # Evaluate
            drift = compute_last_token_kl(
                model, tokenizer,
                baseline_manager.prompts,
                baseline_manager.baseline_logits
            )
            edit_result = {
                "edit_index": i,
                "alpha": 1.0,
                "drift": drift.mean_kl,
            }

        results["per_edit_results"].append(edit_result)

        # Compute cumulative drift
        cumulative_drift = compute_last_token_kl(
            model, tokenizer,
            baseline_manager.prompts,
            baseline_manager.baseline_logits
        )
        results["cumulative_drift"].append(cumulative_drift.mean_kl)

        # Check retention of earlier edits
        retention_scores = []
        for j in range(i + 1):
            prev_request = edit_requests[j]
            prompt = prev_request["prompt"].format(prev_request["subject"])
            target = prev_request["target_new"]["str"]
            success = compute_edit_success(model, tokenizer, prompt, target)
            retention_scores.append({
                "edit_index": j,
                "rank": success.rank_after,
                "retained": success.rank_after <= 10
            })
        results["edit_retention"].append(retention_scores)

    return results
