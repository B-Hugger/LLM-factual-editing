# Preservation-Aware Knowledge Editing for Large Language Models

This repository contains the code for reproducing all experiments from our paper *Preservation-Aware Knowledge Editing for Large Language Models*.

## Overview

Rank-One Model Editing (ROME) is a popular training-free method for updating factual knowledge in LLMs. While effective at installing new facts, ROME's rank-one updates can severely disrupt unrelated generations, producing repetitive or corrupted outputs even when standard drift metrics (KL divergence) suggest everything is fine.

We propose a preservation-aware extension to ROME that explicitly optimizes for retain-set stability before finalizing an edit. The method is lightweight, training-free, and requires only a single hyperparameter λ to control the trade-off between edit success and locality preservation.

## Key Findings

- **ROME is effective but brittle.** Edits reliably install new facts but cause widespread fluency degradation on unrelated prompts.
- **KL divergence underestimates locality failures.** Quantitative metrics stay below acceptable thresholds even when generation quality collapses. Our MEMIT sequential editing experiments show drift metrics improving while edit success degrades.
- **Magnitude scaling alone is insufficient.** Scalar-α reduces drift but does not eliminate qualitative degeneration, indicating the problem is partly structural.
- **Preservation-aware updates help.** Explicitly optimizing for retain-set stability yields more consistent improvements in both metrics and generation quality.

## Experiments

All experiments use GPT-2-small via the official ROME/MEMIT implementations.

| Experiment | Description |
|---|---|
| **Single-fact edits** | Three controlled edits (e.g., "Paris is the capital of Italy") comparing ROME, Scalar-α, and our preservation-aware variant across edit success, locality, and qualitative generation quality. |
| **Scalar-α sweep** | Grid search over α ∈ [0, 1] to isolate the effect of update magnitude on drift and edit success. |
| **MEMIT sequential editing** | 12-step sequential editing protocol measuring cumulative edit success, drift, and edit interference over time. |
| **Preservation-aware ROME** | Evaluation of our λ-controlled variant on the same edit sets, demonstrating reduced drift and improved generation stability. |

## Setup

```bash
# Clone the repository
git clone https://github.com/B-Hugger/LLM-factual-editing.git
cd LLM-factual-editing

# Install dependencies
pip install -r requirements.txt
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{hugger2026preservation,
  title={Preservation-Aware Knowledge Editing for Large Language Models},
  author={Hugger, Brandon},
  year={2026}
}
```

## License

See [LICENSE](LICENSE) for details.