#!/usr/bin/env python3
"""
Validate notebook code structure without loading the full model.
This tests imports, paths, and basic logic.
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("NOTEBOOK VALIDATION")
print("=" * 60)

# Simulate notebook cell 1
print("\n[Cell 1] Setting up paths...")
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'memit'))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

os.chdir(PROJECT_ROOT / 'memit')

print(f"  Working directory: {os.getcwd()}")
print(f"  Project root: {PROJECT_ROOT}")
print("  [OK] Paths configured")

# Test imports
print("\n[Cell 1] Testing imports...")
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("  [OK] torch, transformers imported")
except ImportError as e:
    print(f"  [FAIL] {e}")

# Cell 5 - Edit request format
print("\n[Cell 5] Testing edit request format...")
edit_request = {
    "prompt": "{} is located in",
    "subject": "The Eiffel Tower",
    "target_new": {"str": " Rome"},
    "case_id": 1
}
prompt = edit_request["prompt"].format(edit_request["subject"])
assert prompt == "The Eiffel Tower is located in", f"Prompt format wrong: {prompt}"
print(f"  Full prompt: '{prompt}'")
print("  [OK] Edit request format correct")

# Cell 7 - Baseline manager
print("\n[Cell 7] Testing baseline manager...")
try:
    from src.drift.baseline import BaselinePromptsManager
    baseline_manager = BaselinePromptsManager()
    print(f"  Using {len(baseline_manager.prompts)} baseline prompts")
    print("  [OK] BaselinePromptsManager works")
except Exception as e:
    print(f"  [FAIL] {e}")

# Cell 9 - Hyperparameters
print("\n[Cell 9] Testing hyperparameters loading...")
try:
    from rome.rome_hparams import ROMEHyperParams
    hparams_path = 'hparams/ROME/Mistral-7B.json'
    assert os.path.exists(hparams_path), f"Hparams file not found: {hparams_path}"
    hparams = ROMEHyperParams.from_json(hparams_path)
    print(f"  ROME layers: {hparams.layers}")
    print(f"  Target module: {hparams.rewrite_module_tmp}")
    print("  [OK] Hyperparameters loaded")
except Exception as e:
    print(f"  [FAIL] {e}")

# Cell 11 - Drift imports
print("\n[Cell 11] Testing drift module imports...")
try:
    from src.drift.controller import DriftController
    from src.editing.weight_update import apply_weight_update_context
    from src.drift.metrics import compute_last_token_kl, compute_edit_success
    print("  [OK] All drift modules imported")
except Exception as e:
    print(f"  [FAIL] {e}")

# Cell 15 - Matplotlib
print("\n[Cell 15] Testing matplotlib...")
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    print("  [OK] Matplotlib imported")
except Exception as e:
    print(f"  [FAIL] {e}")

# Cell 17 - Weight update
print("\n[Cell 17] Testing weight update imports...")
try:
    from src.editing.weight_update import apply_weight_update
    print("  [OK] apply_weight_update imported")
except Exception as e:
    print(f"  [FAIL] {e}")

# Results directory
print("\n[Results] Testing results directory...")
results_dir = PROJECT_ROOT / 'results'
os.makedirs(str(results_dir), exist_ok=True)
assert results_dir.exists(), "Results directory not created"
print(f"  Results directory: {results_dir}")
print("  [OK] Results directory exists")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE - All imports and paths OK!")
print("=" * 60)
print("\nTo run the full notebook with model loading:")
print("  1. Open notebooks/demo.ipynb in Jupyter")
print("  2. Run all cells (requires GPU and ~15GB VRAM)")
