"""
Refactored MEMIT module that exposes weight updates (ΔW) without immediate application.

This module provides:
1. execute_memit_exposed: Computes weight deltas for multiple edits without modifying the model
2. Support for batch/multi-edit operations with the MEMIT algorithm

The separation allows for:
- Inspection of deltas before application
- Scaling of updates
- Drift measurement before permanent changes
"""

import sys
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add memit to path
MEMIT_PATH = Path(__file__).parent.parent.parent / "memit"
sys.path.insert(0, str(MEMIT_PATH))

from util import nethook
from util.generate import generate_fast
from util.globals import STATS_DIR
from memit.memit_hparams import MEMITHyperParams
from memit.compute_ks import compute_ks
from memit.compute_z import compute_z, get_module_input_output_at_words
from rome.layer_stats import layer_stats

from .weight_update import WeightUpdate


# Cache variables
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def execute_memit_exposed(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
) -> WeightUpdate:
    """
    Compute MEMIT weight deltas without modifying the model.

    This is a refactored version of the original MEMIT execute_memit function
    that returns weight deltas as a WeightUpdate object instead of
    immediately applying them to the model.

    Args:
        model: The language model
        tok: The tokenizer
        requests: List of edit requests, each containing 'prompt', 'subject', 'target_new'
        hparams: MEMIT hyperparameters
        cache_template: Optional template for caching z vectors

    Returns:
        WeightUpdate containing computed deltas for all layers
    """
    # Prepare requests
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]

    for request in requests[:10]:
        print(
            f"MEMIT request: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Get weight references
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    # Store original weights
    original_weights = {k: v.detach().clone() for k, v in weights.items()}
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z vectors for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request.get("case_id", 0)
                )
            )
            if cache_template is not None
            else None
        )

        data_loaded = False
        if cache_fname is not None and cache_fname.exists():
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(model.device))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache: {e}. Recomputing...")

        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )
            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(cache_fname, v_star=cur_z.detach().cpu().numpy())
                print(f"Cached z vector at {cache_fname}")

    zs = torch.stack(z_list, dim=1)

    # Compute deltas for each layer
    deltas = {}

    for i, layer in enumerate(hparams.layers):
        print(f"\n--- LAYER {layer} ---")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T

        targets = zs - cur_zs
        print(f"z error: {torch.linalg.norm(targets, dim=0).mean():.4f}")

        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Load covariance matrix
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
        )

        # Compute update in double precision
        layer_ks_double = layer_ks.double()
        targets_double = targets.double()

        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov.double() + layer_ks_double @ layer_ks_double.T,
            layer_ks_double,
        )
        resid = targets_double / (len(hparams.layers) - i)  # Distribute residual
        upd_matrix = resid @ adj_k.T

        # Match shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print(f"Original norm: {torch.linalg.norm(weights[weight_name]):.4f}")
        print(f"Update norm: {torch.linalg.norm(upd_matrix):.4f}")

        # Store delta (don't apply to model)
        deltas[weight_name] = upd_matrix.float().detach().clone()

        # Temporarily update weights for next layer's computation
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()

        # Clean up GPU memory
        cov.cpu()
        for x in [layer_ks, layer_ks_double, cur_zs, targets, targets_double]:
            if hasattr(x, 'cpu'):
                x.cpu()
            del x
        torch.cuda.empty_cache()

    # Restore original model weights
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = original_weights[k]

    print(f"\nMEMIT deltas computed for {list(deltas.keys())}")

    return WeightUpdate(
        deltas=deltas,
        method="memit",
        request_info={
            "requests": requests,
            "num_edits": len(requests),
            "layers": hparams.layers
        },
        original_weights=original_weights
    )


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: int,
    mom2_dtype: str,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieve covariance statistics for a layer.
    """
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance for {model_name} @ {layer_name}")

    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return COV_CACHE[key].to(model.device)


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Match update matrix shape to target weight shape."""
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            f"Update matrix shape {matrix.shape} cannot match target shape {shape}"
        )


def get_context_templates(model, tok):
    """Generate context templates for MEMIT computation."""
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]
        ]
        print(f"Cached {sum(len(t) for t in CONTEXT_TEMPLATES_CACHE)} context templates")

    return CONTEXT_TEMPLATES_CACHE


def clear_caches():
    """Clear all caches."""
    global CONTEXT_TEMPLATES_CACHE, COV_CACHE
    CONTEXT_TEMPLATES_CACHE = None
    COV_CACHE = {}
