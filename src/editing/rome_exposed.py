"""
Refactored ROME module that exposes weight updates (ΔW) without immediate application.

This module provides:
1. execute_rome_exposed: Computes weight deltas without modifying the model
2. apply_delta_to_model: Applies computed deltas to a model

The separation allows for:
- Inspection of deltas before application
- Scaling of updates
- Drift measurement before permanent changes
"""

import sys
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add memit to path
MEMIT_PATH = Path(__file__).parent.parent.parent / "memit"
sys.path.insert(0, str(MEMIT_PATH))

from util import nethook
from util.generate import generate_fast
from rome.rome_hparams import ROMEHyperParams
from rome.compute_u import compute_u
from rome.compute_v import compute_v

from .weight_update import WeightUpdate, compute_update_matrix


# Cache for context templates
CONTEXT_TEMPLATES_CACHE = None


def execute_rome_exposed(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    cache_template: Optional[str] = None,
) -> WeightUpdate:
    """
    Compute ROME weight deltas without modifying the model.

    This is a refactored version of the original ROME execute_rome function
    that returns weight deltas as a WeightUpdate object instead of
    immediately applying them to the model.

    Args:
        model: The language model
        tok: The tokenizer
        request: Edit request containing 'prompt', 'subject', 'target_new'
        hparams: ROME hyperparameters
        cache_template: Optional template for caching k/v pairs

    Returns:
        WeightUpdate containing computed deltas
    """
    # Prepare request
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":
        request["target_new"]["str"] = " " + request["target_new"]["str"]

    print(
        f"Computing ROME deltas for: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
    )

    # Get weight references
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    # Store original weights for potential restoration
    original_weights = {k: v.detach().clone() for k, v in weights.items()}

    # Compute deltas for each layer
    deltas = {}
    hparams_layers = sorted(hparams.layers)

    for layer in hparams_layers:
        left_vector, right_vector = None, None
        require_recompute = True

        # Check cache for first layer
        if layer == hparams_layers[0]:
            cache_fname = (
                Path(
                    str(cache_template).format(
                        layer, hparams.clamp_norm_factor, request.get("case_id", 0)
                    )
                )
                if cache_template is not None
                else None
            )
            if cache_fname is not None and cache_fname.exists():
                try:
                    data = np.load(cache_fname)
                    left_vector = torch.from_numpy(data["left_vector"]).to(model.device)
                    right_vector = torch.from_numpy(data["right_vector"]).to(model.device)
                    require_recompute = False
                except Exception as e:
                    print(f"Error reading cache: {e}. Recomputing...")

        # Compute left vector (u)
        if left_vector is None:
            left_vector = compute_u(
                model,
                tok,
                request,
                hparams,
                layer,
                get_context_templates(model, tok, hparams.context_template_length_params),
            )
        print(f"Left vector shape: {left_vector.shape}")

        # Compute right vector (v)
        if right_vector is None:
            right_vector = compute_v(
                model,
                tok,
                request,
                hparams,
                layer,
                left_vector,
                get_context_templates(model, tok, hparams.context_template_length_params),
            )
        print(f"Right vector shape: {right_vector.shape}")

        # Cache vectors if template provided
        if cache_fname is not None and require_recompute:
            cache_fname.parent.mkdir(exist_ok=True, parents=True)
            np.savez(
                cache_fname,
                left_vector=left_vector.detach().cpu().numpy(),
                right_vector=right_vector.detach().cpu().numpy(),
            )
            print(f"Cached k/v pair at {cache_fname}")

        # Compute update matrix
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = compute_update_matrix(
            left_vector,
            right_vector,
            weights[weight_name].shape
        )

        deltas[weight_name] = upd_matrix.detach().clone()

    print(f"ROME deltas computed for {list(deltas.keys())}")

    return WeightUpdate(
        deltas=deltas,
        method="rome",
        request_info={"request": request, "hparams_layers": hparams_layers},
        original_weights=original_weights
    )


def apply_delta_to_model(
    model: AutoModelForCausalLM,
    weight_update: WeightUpdate,
    alpha: float = 1.0
) -> AutoModelForCausalLM:
    """
    Apply weight deltas to a model with optional scaling.

    Args:
        model: The model to modify
        weight_update: WeightUpdate containing deltas
        alpha: Scaling factor (1.0 = full update)

    Returns:
        The modified model (same object, modified in place)
    """
    with torch.no_grad():
        for w_name, delta in weight_update.deltas.items():
            w = nethook.get_parameter(model, w_name)
            scaled_delta = delta.to(w.device).to(w.dtype) * alpha
            w[...] += scaled_delta

    print(f"Applied {'scaled ' if alpha != 1.0 else ''}deltas to {list(weight_update.deltas.keys())}")
    return model


def get_context_templates(model, tok, length_params):
    """
    Generate context templates for ROME computation.
    Caches results for efficiency.
    """
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["<|endoftext|>"] if hasattr(tok, 'eos_token') and tok.eos_token == "<|endoftext|>"
                        else [tok.bos_token if tok.bos_token else ""],
                        n_gen_per_prompt=n_gen,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]
        print(f"Cached context templates: {len(CONTEXT_TEMPLATES_CACHE)} templates")

    return CONTEXT_TEMPLATES_CACHE


def clear_context_cache():
    """Clear the context template cache."""
    global CONTEXT_TEMPLATES_CACHE
    CONTEXT_TEMPLATES_CACHE = None
