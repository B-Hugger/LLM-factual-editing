"""
Weight Update Module
Provides data structures and utilities for managing weight deltas (ΔW)
independently from immediate application to model weights.
"""

import contextlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
import torch
from copy import deepcopy


@dataclass
class WeightUpdate:
    """
    Container for weight updates computed by ROME or MEMIT.

    Attributes:
        deltas: Dictionary mapping parameter names to update tensors
        method: The editing method used ('rome' or 'memit')
        request_info: Information about the edit request(s)
        original_weights: Backup of original weights for restoration
    """
    deltas: Dict[str, torch.Tensor] = field(default_factory=dict)
    method: str = ""
    request_info: Dict = field(default_factory=dict)
    original_weights: Dict[str, torch.Tensor] = field(default_factory=dict)

    def scale(self, alpha: float) -> 'WeightUpdate':
        """
        Return a new WeightUpdate with scaled deltas.

        Args:
            alpha: Scaling factor

        Returns:
            New WeightUpdate with scaled deltas
        """
        scaled_deltas = {k: v * alpha for k, v in self.deltas.items()}
        return WeightUpdate(
            deltas=scaled_deltas,
            method=self.method,
            request_info=self.request_info,
            original_weights=self.original_weights
        )

    def to(self, device: Union[str, torch.device]) -> 'WeightUpdate':
        """Move all tensors to specified device."""
        moved_deltas = {k: v.to(device) for k, v in self.deltas.items()}
        moved_orig = {k: v.to(device) for k, v in self.original_weights.items()}
        return WeightUpdate(
            deltas=moved_deltas,
            method=self.method,
            request_info=self.request_info,
            original_weights=moved_orig
        )

    @property
    def total_norm(self) -> float:
        """Compute the total Frobenius norm of all deltas."""
        total = 0.0
        for delta in self.deltas.values():
            total += torch.norm(delta).item() ** 2
        return total ** 0.5

    def __repr__(self):
        return (f"WeightUpdate(method={self.method}, "
                f"num_params={len(self.deltas)}, "
                f"total_norm={self.total_norm:.4f})")


def get_parameter(model, name: str) -> torch.nn.Parameter:
    """
    Get a parameter from a model by its dotted name.

    Args:
        model: The PyTorch model
        name: Dotted parameter name (e.g., 'model.layers.5.mlp.down_proj.weight')

    Returns:
        The parameter tensor
    """
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(f"Parameter {name} not found in model")


def apply_weight_update(
    model: torch.nn.Module,
    weight_update: WeightUpdate,
    alpha: float = 1.0,
    inplace: bool = True
) -> torch.nn.Module:
    """
    Permanently apply a weight update to a model.

    Args:
        model: The model to modify
        weight_update: The WeightUpdate containing deltas
        alpha: Scaling factor for the update
        inplace: Whether to modify the model in-place

    Returns:
        The modified model
    """
    if not inplace:
        model = deepcopy(model)

    with torch.no_grad():
        for param_name, delta in weight_update.deltas.items():
            param = get_parameter(model, param_name)
            delta_scaled = delta.to(param.device).to(param.dtype) * alpha
            param.add_(delta_scaled)

    return model


@contextlib.contextmanager
def apply_weight_update_context(
    model: torch.nn.Module,
    weight_update: WeightUpdate,
    alpha: float = 1.0
):
    """
    Context manager for temporarily applying a weight update.
    Restores original weights upon exit.

    Args:
        model: The model to modify
        weight_update: The WeightUpdate containing deltas
        alpha: Scaling factor for the update

    Yields:
        The temporarily modified model

    Example:
        with apply_weight_update_context(model, update, alpha=0.5) as temp_model:
            # temp_model has the scaled update applied
            outputs = temp_model(inputs)
        # Original weights are restored here
    """
    # Store original weights
    original_weights = {}
    for param_name in weight_update.deltas.keys():
        try:
            param = get_parameter(model, param_name)
            original_weights[param_name] = param.detach().clone()
        except LookupError:
            print(f"Warning: Parameter {param_name} not found, skipping")

    # Apply the scaled update
    with torch.no_grad():
        for param_name, delta in weight_update.deltas.items():
            if param_name in original_weights:
                param = get_parameter(model, param_name)
                delta_scaled = delta.to(param.device).to(param.dtype) * alpha
                param.add_(delta_scaled)

    try:
        yield model
    finally:
        # Restore original weights
        with torch.no_grad():
            for param_name, orig_weight in original_weights.items():
                param = get_parameter(model, param_name)
                param.copy_(orig_weight)


def compute_update_matrix(
    left_vector: torch.Tensor,
    right_vector: torch.Tensor,
    target_shape: torch.Size
) -> torch.Tensor:
    """
    Compute the rank-1 update matrix from left and right vectors.
    Handles transposition to match target weight shape.

    Args:
        left_vector: The 'u' or 'key' vector
        right_vector: The 'v' or 'value' vector
        target_shape: The shape of the target weight matrix

    Returns:
        Update matrix matching target_shape
    """
    # Compute outer product
    upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)

    # Match shape (handle transposed weights)
    if upd_matrix.shape == target_shape:
        return upd_matrix
    elif upd_matrix.T.shape == target_shape:
        return upd_matrix.T
    else:
        raise ValueError(
            f"Update matrix shape {upd_matrix.shape} cannot match "
            f"target shape {target_shape}"
        )
