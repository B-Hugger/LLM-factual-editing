"""
Experiment Runner for Stabilized Model Editing

Provides functions to run single-edit ROME and multi-edit MEMIT experiments,
comparing baseline vs stabilized editing approaches.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "memit"))

from rome.rome_hparams import ROMEHyperParams
from memit.memit_hparams import MEMITHyperParams

from ..editing.rome_exposed import execute_rome_exposed
from ..editing.memit_exposed import execute_memit_exposed
from ..editing.weight_update import apply_weight_update, WeightUpdate
from ..drift.baseline import BaselinePromptsManager
from ..drift.controller import DriftController, OptimizationResult
from ..drift.metrics import compute_last_token_kl, compute_edit_success


# Sample edit requests for testing
SAMPLE_EDIT_REQUESTS = [
    {
        "prompt": "The Eiffel Tower is located in",
        "subject": "The Eiffel Tower",
        "target_new": {"str": " Rome"},
        "case_id": 1
    },
    {
        "prompt": "The CEO of Apple is",
        "subject": "Apple",
        "target_new": {"str": " Elon Musk"},
        "case_id": 2
    },
    {
        "prompt": "The capital of France is",
        "subject": "France",
        "target_new": {"str": " Berlin"},
        "case_id": 3
    },
    {
        "prompt": "Microsoft was founded by",
        "subject": "Microsoft",
        "target_new": {"str": " Steve Jobs"},
        "case_id": 4
    },
    {
        "prompt": "The Great Wall of China is in",
        "subject": "The Great Wall of China",
        "target_new": {"str": " Japan"},
        "case_id": 5
    },
]


def run_single_edit_experiment(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    hparams: ROMEHyperParams,
    edit_request: Dict,
    baseline_manager: BaselinePromptsManager,
    drift_threshold: float = 0.1,
    results_dir: str = "results",
    device: str = "cuda"
) -> Dict:
    """
    Run a single-edit ROME experiment comparing baseline vs stabilized.

    Args:
        model: The language model
        tokenizer: The tokenizer
        hparams: ROME hyperparameters
        edit_request: The edit request
        baseline_manager: Manager with baseline prompts and logits
        drift_threshold: Threshold for drift control
        results_dir: Directory to save results
        device: Device to use

    Returns:
        Dictionary with experiment results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "method": "ROME",
        "edit_request": edit_request,
        "baseline_results": {},
        "stabilized_results": {},
    }

    # Ensure baselines are computed
    if not baseline_manager.has_baselines():
        baseline_manager.compute_baselines(model, tokenizer, device)

    print("=" * 60)
    print("SINGLE-EDIT ROME EXPERIMENT")
    print("=" * 60)
    print(f"Edit: {edit_request['prompt'].format(edit_request['subject'])} -> {edit_request['target_new']['str']}")

    # ===== BASELINE (full update, alpha=1.0) =====
    print("\n--- BASELINE (alpha=1.0) ---")

    # Compute weight update
    weight_update = execute_rome_exposed(model, tokenizer, edit_request, hparams)
    print(f"Update norm: {weight_update.total_norm:.4f}")

    # Measure pre-edit metrics
    prompt = edit_request["prompt"].format(edit_request["subject"])
    target = edit_request["target_new"]["str"]
    pre_edit_success = compute_edit_success(model, tokenizer, prompt, target, device=device)
    pre_edit_drift = compute_last_token_kl(
        model, tokenizer, baseline_manager.prompts,
        baseline_manager.baseline_logits, device
    )

    # Apply full update
    model_baseline = apply_weight_update(model, weight_update, alpha=1.0, inplace=False)

    # Measure post-edit metrics
    post_edit_success_baseline = compute_edit_success(
        model_baseline, tokenizer, prompt, target, device=device
    )
    post_edit_drift_baseline = compute_last_token_kl(
        model_baseline, tokenizer, baseline_manager.prompts,
        baseline_manager.baseline_logits, device
    )

    results["baseline_results"] = {
        "alpha": 1.0,
        "pre_edit": {
            "target_log_prob": pre_edit_success.target_log_prob_after,
            "target_rank": pre_edit_success.rank_after,
        },
        "post_edit": {
            "target_log_prob": post_edit_success_baseline.target_log_prob_after,
            "target_rank": post_edit_success_baseline.rank_after,
            "log_prob_improvement": post_edit_success_baseline.target_log_prob_after - pre_edit_success.target_log_prob_after,
        },
        "drift": {
            "mean_kl": post_edit_drift_baseline.mean_kl,
            "max_kl": post_edit_drift_baseline.max_kl,
        }
    }

    print(f"  Pre-edit rank: {pre_edit_success.rank_after}")
    print(f"  Post-edit rank: {post_edit_success_baseline.rank_after}")
    print(f"  Log prob improvement: {results['baseline_results']['post_edit']['log_prob_improvement']:.4f}")
    print(f"  Mean drift (KL): {post_edit_drift_baseline.mean_kl:.6f}")

    # Clean up baseline model
    del model_baseline
    torch.cuda.empty_cache()

    # ===== STABILIZED (optimized alpha) =====
    print("\n--- STABILIZED (optimized alpha) ---")

    controller = DriftController(
        baseline_manager=baseline_manager,
        drift_threshold=drift_threshold,
        device=device
    )

    # Recompute weight update (or reuse)
    weight_update = execute_rome_exposed(model, tokenizer, edit_request, hparams)

    # Find optimal alpha
    opt_result = controller.find_optimal_alpha(
        model, tokenizer, weight_update, edit_request
    )

    # Apply stabilized update
    model_stabilized = apply_weight_update(
        model, weight_update, alpha=opt_result.optimal_alpha, inplace=False
    )

    # Measure final metrics
    post_edit_success_stabilized = compute_edit_success(
        model_stabilized, tokenizer, prompt, target, device=device
    )
    post_edit_drift_stabilized = compute_last_token_kl(
        model_stabilized, tokenizer, baseline_manager.prompts,
        baseline_manager.baseline_logits, device
    )

    results["stabilized_results"] = {
        "alpha": opt_result.optimal_alpha,
        "converged": opt_result.converged,
        "post_edit": {
            "target_log_prob": post_edit_success_stabilized.target_log_prob_after,
            "target_rank": post_edit_success_stabilized.rank_after,
            "log_prob_improvement": post_edit_success_stabilized.target_log_prob_after - pre_edit_success.target_log_prob_after,
        },
        "drift": {
            "mean_kl": post_edit_drift_stabilized.mean_kl,
            "max_kl": post_edit_drift_stabilized.max_kl,
        },
        "search_history": opt_result.search_history
    }

    print(f"  Optimal alpha: {opt_result.optimal_alpha:.4f}")
    print(f"  Post-edit rank: {post_edit_success_stabilized.rank_after}")
    print(f"  Log prob improvement: {results['stabilized_results']['post_edit']['log_prob_improvement']:.4f}")
    print(f"  Mean drift (KL): {post_edit_drift_stabilized.mean_kl:.6f}")

    # Summary comparison
    print("\n--- SUMMARY ---")
    drift_reduction = (
        (post_edit_drift_baseline.mean_kl - post_edit_drift_stabilized.mean_kl)
        / post_edit_drift_baseline.mean_kl * 100
        if post_edit_drift_baseline.mean_kl > 0 else 0
    )
    print(f"  Drift reduction: {drift_reduction:.1f}%")

    # Clean up
    del model_stabilized
    torch.cuda.empty_cache()

    # Save results
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f"single_edit_{edit_request['case_id']}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def run_multi_edit_experiment(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    hparams: MEMITHyperParams,
    edit_requests: List[Dict],
    baseline_manager: BaselinePromptsManager,
    drift_threshold: float = 0.1,
    results_dir: str = "results",
    device: str = "cuda"
) -> Dict:
    """
    Run a multi-edit MEMIT experiment comparing baseline vs stabilized.

    Args:
        model: The language model
        tokenizer: The tokenizer
        hparams: MEMIT hyperparameters
        edit_requests: List of edit requests
        baseline_manager: Manager with baseline prompts and logits
        drift_threshold: Threshold for drift control
        results_dir: Directory to save results
        device: Device to use

    Returns:
        Dictionary with experiment results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "method": "MEMIT",
        "num_edits": len(edit_requests),
        "edit_requests": edit_requests,
        "baseline_results": {},
        "stabilized_results": {},
    }

    # Ensure baselines are computed
    if not baseline_manager.has_baselines():
        baseline_manager.compute_baselines(model, tokenizer, device)

    print("=" * 60)
    print(f"MULTI-EDIT MEMIT EXPERIMENT ({len(edit_requests)} edits)")
    print("=" * 60)

    for req in edit_requests:
        print(f"  - {req['prompt'].format(req['subject'])} -> {req['target_new']['str']}")

    # Compute pre-edit metrics for all targets
    pre_edit_metrics = []
    for req in edit_requests:
        prompt = req["prompt"].format(req["subject"])
        target = req["target_new"]["str"]
        success = compute_edit_success(model, tokenizer, prompt, target, device=device)
        pre_edit_metrics.append({
            "prompt": prompt,
            "target": target,
            "log_prob": success.target_log_prob_after,
            "rank": success.rank_after
        })

    # ===== BASELINE (full update, alpha=1.0) =====
    print("\n--- BASELINE (alpha=1.0) ---")

    # Compute weight update
    weight_update = execute_memit_exposed(model, tokenizer, edit_requests, hparams)
    print(f"Update norm: {weight_update.total_norm:.4f}")

    # Apply full update
    model_baseline = apply_weight_update(model, weight_update, alpha=1.0, inplace=False)

    # Measure post-edit metrics
    post_edit_baseline = []
    for i, req in enumerate(edit_requests):
        prompt = req["prompt"].format(req["subject"])
        target = req["target_new"]["str"]
        success = compute_edit_success(model_baseline, tokenizer, prompt, target, device=device)
        post_edit_baseline.append({
            "prompt": prompt,
            "target": target,
            "log_prob": success.target_log_prob_after,
            "rank": success.rank_after,
            "improvement": success.target_log_prob_after - pre_edit_metrics[i]["log_prob"]
        })

    drift_baseline = compute_last_token_kl(
        model_baseline, tokenizer, baseline_manager.prompts,
        baseline_manager.baseline_logits, device
    )

    results["baseline_results"] = {
        "alpha": 1.0,
        "pre_edit": pre_edit_metrics,
        "post_edit": post_edit_baseline,
        "drift": {
            "mean_kl": drift_baseline.mean_kl,
            "max_kl": drift_baseline.max_kl,
        },
        "success_rate": sum(1 for m in post_edit_baseline if m["rank"] <= 10) / len(post_edit_baseline)
    }

    print(f"  Mean drift (KL): {drift_baseline.mean_kl:.6f}")
    print(f"  Success rate (rank <= 10): {results['baseline_results']['success_rate']:.2%}")

    del model_baseline
    torch.cuda.empty_cache()

    # ===== STABILIZED (optimized alpha) =====
    print("\n--- STABILIZED (optimized alpha) ---")

    # For multi-edit, we use a single representative request for optimization
    # (or average over all requests)
    controller = DriftController(
        baseline_manager=baseline_manager,
        drift_threshold=drift_threshold,
        device=device
    )

    # Recompute weight update
    weight_update = execute_memit_exposed(model, tokenizer, edit_requests, hparams)

    # Find optimal alpha using first request as representative
    opt_result = controller.find_optimal_alpha(
        model, tokenizer, weight_update, edit_requests[0]
    )

    # Apply stabilized update
    model_stabilized = apply_weight_update(
        model, weight_update, alpha=opt_result.optimal_alpha, inplace=False
    )

    # Measure post-edit metrics
    post_edit_stabilized = []
    for i, req in enumerate(edit_requests):
        prompt = req["prompt"].format(req["subject"])
        target = req["target_new"]["str"]
        success = compute_edit_success(model_stabilized, tokenizer, prompt, target, device=device)
        post_edit_stabilized.append({
            "prompt": prompt,
            "target": target,
            "log_prob": success.target_log_prob_after,
            "rank": success.rank_after,
            "improvement": success.target_log_prob_after - pre_edit_metrics[i]["log_prob"]
        })

    drift_stabilized = compute_last_token_kl(
        model_stabilized, tokenizer, baseline_manager.prompts,
        baseline_manager.baseline_logits, device
    )

    results["stabilized_results"] = {
        "alpha": opt_result.optimal_alpha,
        "converged": opt_result.converged,
        "post_edit": post_edit_stabilized,
        "drift": {
            "mean_kl": drift_stabilized.mean_kl,
            "max_kl": drift_stabilized.max_kl,
        },
        "success_rate": sum(1 for m in post_edit_stabilized if m["rank"] <= 10) / len(post_edit_stabilized),
        "search_history": opt_result.search_history
    }

    print(f"  Optimal alpha: {opt_result.optimal_alpha:.4f}")
    print(f"  Mean drift (KL): {drift_stabilized.mean_kl:.6f}")
    print(f"  Success rate (rank <= 10): {results['stabilized_results']['success_rate']:.2%}")

    # Summary
    print("\n--- SUMMARY ---")
    if drift_baseline.mean_kl > 0:
        drift_reduction = (drift_baseline.mean_kl - drift_stabilized.mean_kl) / drift_baseline.mean_kl * 100
        print(f"  Drift reduction: {drift_reduction:.1f}%")

    del model_stabilized
    torch.cuda.empty_cache()

    # Save results
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f"multi_edit_{len(edit_requests)}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def run_sequential_editing_experiment(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    hparams: ROMEHyperParams,
    edit_requests: List[Dict],
    baseline_manager: BaselinePromptsManager,
    drift_threshold: float = 0.1,
    results_dir: str = "results",
    device: str = "cuda"
) -> Dict:
    """
    Run sequential editing experiment: apply multiple edits one at a time.

    Tracks:
    - Edit retention over time
    - Cumulative drift
    - Performance degradation

    Args:
        model: The language model
        tokenizer: The tokenizer
        hparams: ROME hyperparameters
        edit_requests: List of edit requests to apply sequentially
        baseline_manager: Manager with baseline prompts
        drift_threshold: Threshold for drift control
        results_dir: Directory to save results
        device: Device to use

    Returns:
        Dictionary with detailed sequential results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "method": "ROME_Sequential",
        "num_edits": len(edit_requests),
        "baseline_sequential": [],
        "stabilized_sequential": [],
    }

    if not baseline_manager.has_baselines():
        baseline_manager.compute_baselines(model, tokenizer, device)

    print("=" * 60)
    print(f"SEQUENTIAL EDITING EXPERIMENT ({len(edit_requests)} edits)")
    print("=" * 60)

    # ===== BASELINE SEQUENTIAL =====
    print("\n--- BASELINE SEQUENTIAL ---")
    model_baseline = model  # Will modify in place

    for i, req in enumerate(edit_requests):
        print(f"\nEdit {i+1}/{len(edit_requests)}: {req['prompt'].format(req['subject'])} -> {req['target_new']['str']}")

        # Compute and apply update
        weight_update = execute_rome_exposed(model_baseline, tokenizer, req, hparams)
        model_baseline = apply_weight_update(model_baseline, weight_update, alpha=1.0, inplace=False)

        # Measure drift
        drift = compute_last_token_kl(
            model_baseline, tokenizer, baseline_manager.prompts,
            baseline_manager.baseline_logits, device
        )

        # Measure retention of all previous edits
        retention = []
        for j in range(i + 1):
            prev_req = edit_requests[j]
            prompt = prev_req["prompt"].format(prev_req["subject"])
            target = prev_req["target_new"]["str"]
            success = compute_edit_success(model_baseline, tokenizer, prompt, target, device=device)
            retention.append({
                "edit_index": j,
                "rank": success.rank_after,
                "retained": success.rank_after <= 10
            })

        results["baseline_sequential"].append({
            "edit_index": i,
            "drift_mean_kl": drift.mean_kl,
            "retention": retention,
            "retention_rate": sum(1 for r in retention if r["retained"]) / len(retention)
        })

        print(f"  Drift: {drift.mean_kl:.6f}, Retention: {results['baseline_sequential'][-1]['retention_rate']:.2%}")

    del model_baseline
    torch.cuda.empty_cache()

    # ===== STABILIZED SEQUENTIAL =====
    print("\n--- STABILIZED SEQUENTIAL ---")
    model_stabilized = model

    controller = DriftController(
        baseline_manager=baseline_manager,
        drift_threshold=drift_threshold,
        device=device
    )

    for i, req in enumerate(edit_requests):
        print(f"\nEdit {i+1}/{len(edit_requests)}: {req['prompt'].format(req['subject'])} -> {req['target_new']['str']}")

        # Compute update
        weight_update = execute_rome_exposed(model_stabilized, tokenizer, req, hparams)

        # Find optimal alpha
        opt_result = controller.find_optimal_alpha(
            model_stabilized, tokenizer, weight_update, req
        )

        # Apply stabilized update
        model_stabilized = apply_weight_update(
            model_stabilized, weight_update, alpha=opt_result.optimal_alpha, inplace=False
        )

        # Measure drift
        drift = compute_last_token_kl(
            model_stabilized, tokenizer, baseline_manager.prompts,
            baseline_manager.baseline_logits, device
        )

        # Measure retention
        retention = []
        for j in range(i + 1):
            prev_req = edit_requests[j]
            prompt = prev_req["prompt"].format(prev_req["subject"])
            target = prev_req["target_new"]["str"]
            success = compute_edit_success(model_stabilized, tokenizer, prompt, target, device=device)
            retention.append({
                "edit_index": j,
                "rank": success.rank_after,
                "retained": success.rank_after <= 10
            })

        results["stabilized_sequential"].append({
            "edit_index": i,
            "alpha": opt_result.optimal_alpha,
            "drift_mean_kl": drift.mean_kl,
            "retention": retention,
            "retention_rate": sum(1 for r in retention if r["retained"]) / len(retention)
        })

        print(f"  Alpha: {opt_result.optimal_alpha:.4f}, Drift: {drift.mean_kl:.6f}, Retention: {results['stabilized_sequential'][-1]['retention_rate']:.2%}")

    # Save results
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f"sequential_{len(edit_requests)}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results
