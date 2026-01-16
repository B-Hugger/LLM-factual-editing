"""
Visualization Module

Provides plotting functions for experiment results:
- Drift comparison between baseline and stabilized
- Sequential editing analysis
- Alpha optimization curves
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_drift_comparison(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5)
) -> plt.Figure:
    """
    Plot drift comparison between baseline and stabilized editing.

    Args:
        results: Dictionary with 'baseline_results' and 'stabilized_results'
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Get data
    baseline = results.get("baseline_results", {})
    stabilized = results.get("stabilized_results", {})

    # Plot 1: Drift comparison (bar chart)
    ax1 = axes[0]
    methods = ["Baseline\n(α=1.0)", f"Stabilized\n(α={stabilized.get('alpha', 'N/A'):.3f})"]
    drift_values = [
        baseline.get("drift", {}).get("mean_kl", 0),
        stabilized.get("drift", {}).get("mean_kl", 0)
    ]

    bars = ax1.bar(methods, drift_values, color=['#e74c3c', '#27ae60'], edgecolor='black')
    ax1.set_ylabel("Mean KL Divergence (Drift)")
    ax1.set_title("Drift Comparison")
    ax1.set_ylim(0, max(drift_values) * 1.2 if max(drift_values) > 0 else 1)

    # Add value labels on bars
    for bar, val in zip(bars, drift_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # Plot 2: Alpha search history (if available)
    ax2 = axes[1]
    search_history = stabilized.get("search_history", [])

    if search_history:
        iterations = [h["iteration"] for h in search_history]
        alphas = [h["alpha"] for h in search_history]
        kls = [h["mean_kl"] for h in search_history]

        ax2_twin = ax2.twinx()

        line1 = ax2.plot(iterations, alphas, 'b-o', label='Alpha', linewidth=2)
        line2 = ax2_twin.plot(iterations, kls, 'r-s', label='Mean KL', linewidth=2)

        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Alpha (scaling factor)", color='blue')
        ax2_twin.set_ylabel("Mean KL Divergence", color='red')

        # Add drift threshold line
        drift_threshold = 0.1  # Default
        ax2_twin.axhline(y=drift_threshold, color='red', linestyle='--',
                         alpha=0.5, label='Drift Threshold')

        ax2.set_title("Binary Search Optimization")
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2_twin.tick_params(axis='y', labelcolor='red')

        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right')
    else:
        ax2.text(0.5, 0.5, "No search history available",
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Binary Search Optimization")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_sequential_editing_results(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10)
) -> plt.Figure:
    """
    Plot sequential editing results showing drift accumulation and retention.

    Args:
        results: Dictionary with sequential editing results
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    baseline_seq = results.get("baseline_sequential", [])
    stabilized_seq = results.get("stabilized_sequential", [])

    if not baseline_seq or not stabilized_seq:
        print("No sequential results to plot")
        return fig

    num_edits = len(baseline_seq)
    edit_indices = list(range(1, num_edits + 1))

    # Plot 1: Cumulative drift comparison
    ax1 = axes[0, 0]
    baseline_drift = [r["drift_mean_kl"] for r in baseline_seq]
    stabilized_drift = [r["drift_mean_kl"] for r in stabilized_seq]

    ax1.plot(edit_indices, baseline_drift, 'r-o', label='Baseline (α=1.0)',
             linewidth=2, markersize=8)
    ax1.plot(edit_indices, stabilized_drift, 'g-s', label='Stabilized',
             linewidth=2, markersize=8)

    ax1.set_xlabel("Number of Sequential Edits")
    ax1.set_ylabel("Cumulative Mean KL Divergence")
    ax1.set_title("Drift Accumulation Over Sequential Edits")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Alpha values used (stabilized only)
    ax2 = axes[0, 1]
    if "alpha" in stabilized_seq[0]:
        alphas = [r["alpha"] for r in stabilized_seq]
        ax2.bar(edit_indices, alphas, color='#3498db', edgecolor='black')
        ax2.set_xlabel("Edit Index")
        ax2.set_ylabel("Alpha (scaling factor)")
        ax2.set_title("Scaling Factors Used in Stabilized Editing")
        ax2.set_ylim(0, 1.1)

        # Add mean alpha line
        mean_alpha = np.mean(alphas)
        ax2.axhline(y=mean_alpha, color='red', linestyle='--',
                    label=f'Mean α = {mean_alpha:.3f}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No alpha data available",
                 ha='center', va='center', transform=ax2.transAxes)

    ax2.grid(True, alpha=0.3)

    # Plot 3: Edit retention rates
    ax3 = axes[1, 0]
    baseline_retention = [r["retention_rate"] for r in baseline_seq]
    stabilized_retention = [r["retention_rate"] for r in stabilized_seq]

    ax3.plot(edit_indices, baseline_retention, 'r-o', label='Baseline',
             linewidth=2, markersize=8)
    ax3.plot(edit_indices, stabilized_retention, 'g-s', label='Stabilized',
             linewidth=2, markersize=8)

    ax3.set_xlabel("Number of Sequential Edits")
    ax3.set_ylabel("Edit Retention Rate")
    ax3.set_title("Retention of Previous Edits")
    ax3.set_ylim(0, 1.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Final retention breakdown
    ax4 = axes[1, 1]

    # Get final retention status for each edit
    if baseline_seq and stabilized_seq:
        final_baseline_retention = baseline_seq[-1].get("retention", [])
        final_stabilized_retention = stabilized_seq[-1].get("retention", [])

        if final_baseline_retention:
            baseline_retained = [1 if r["retained"] else 0 for r in final_baseline_retention]
            stabilized_retained = [1 if r["retained"] else 0 for r in final_stabilized_retention]

            x = np.arange(len(baseline_retained))
            width = 0.35

            ax4.bar(x - width/2, baseline_retained, width, label='Baseline',
                    color='#e74c3c', alpha=0.8)
            ax4.bar(x + width/2, stabilized_retained, width, label='Stabilized',
                    color='#27ae60', alpha=0.8)

            ax4.set_xlabel("Edit Index")
            ax4.set_ylabel("Retained (1) / Lost (0)")
            ax4.set_title("Final Edit Retention Status")
            ax4.set_xticks(x)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, "No retention data available",
                     ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, "No data available",
                 ha='center', va='center', transform=ax4.transAxes)

    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_multi_edit_results(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Plot multi-edit (MEMIT) results.

    Args:
        results: Dictionary with multi-edit results
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    baseline = results.get("baseline_results", {})
    stabilized = results.get("stabilized_results", {})

    # Plot 1: Per-edit success (rank)
    ax1 = axes[0]
    if baseline.get("post_edit") and stabilized.get("post_edit"):
        n_edits = len(baseline["post_edit"])
        x = np.arange(n_edits)
        width = 0.35

        baseline_ranks = [r["rank"] for r in baseline["post_edit"]]
        stabilized_ranks = [r["rank"] for r in stabilized["post_edit"]]

        ax1.bar(x - width/2, baseline_ranks, width, label='Baseline',
                color='#e74c3c', alpha=0.8)
        ax1.bar(x + width/2, stabilized_ranks, width, label='Stabilized',
                color='#27ae60', alpha=0.8)

        ax1.axhline(y=10, color='black', linestyle='--', alpha=0.5,
                    label='Success threshold')

        ax1.set_xlabel("Edit Index")
        ax1.set_ylabel("Target Token Rank")
        ax1.set_title("Edit Success (Lower is Better)")
        ax1.set_yscale('log')
        ax1.legend()
        ax1.set_xticks(x)
    else:
        ax1.text(0.5, 0.5, "No edit data", ha='center', va='center',
                 transform=ax1.transAxes)

    ax1.grid(True, alpha=0.3)

    # Plot 2: Overall metrics comparison
    ax2 = axes[1]
    metrics = ['Drift\n(Mean KL)', 'Success Rate', 'Alpha']
    baseline_vals = [
        baseline.get("drift", {}).get("mean_kl", 0),
        baseline.get("success_rate", 0),
        baseline.get("alpha", 1.0)
    ]
    stabilized_vals = [
        stabilized.get("drift", {}).get("mean_kl", 0),
        stabilized.get("success_rate", 0),
        stabilized.get("alpha", 1.0)
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax2.bar(x - width/2, baseline_vals, width, label='Baseline', color='#e74c3c')
    ax2.bar(x + width/2, stabilized_vals, width, label='Stabilized', color='#27ae60')

    ax2.set_ylabel("Value")
    ax2.set_title("Overall Metrics Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Log probability improvements
    ax3 = axes[2]
    if baseline.get("post_edit") and stabilized.get("post_edit"):
        baseline_impr = [r.get("improvement", 0) for r in baseline["post_edit"]]
        stabilized_impr = [r.get("improvement", 0) for r in stabilized["post_edit"]]

        ax3.bar(x - width/2, baseline_impr, width, label='Baseline', color='#e74c3c')
        ax3.bar(x + width/2, stabilized_impr, width, label='Stabilized', color='#27ae60')

        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        ax3.set_xlabel("Edit Index")
        ax3.set_ylabel("Log Probability Improvement")
        ax3.set_title("Edit Effectiveness")
        ax3.legend()
        ax3.set_xticks(x)
    else:
        ax3.text(0.5, 0.5, "No improvement data", ha='center', va='center',
                 transform=ax3.transAxes)

    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def generate_summary_report(
    results_dir: str,
    output_path: str = "summary_report.txt"
) -> str:
    """
    Generate a text summary report from all results in a directory.

    Args:
        results_dir: Directory containing result JSON files
        output_path: Path to save the report

    Returns:
        Summary report as string
    """
    results_path = Path(results_dir)
    report_lines = [
        "=" * 60,
        "STABILIZED MODEL EDITING - SUMMARY REPORT",
        "=" * 60,
        ""
    ]

    # Find all result files
    result_files = list(results_path.glob("*.json"))

    if not result_files:
        report_lines.append("No result files found.")
        return "\n".join(report_lines)

    for result_file in sorted(result_files):
        with open(result_file, "r") as f:
            results = json.load(f)

        report_lines.append(f"\n--- {result_file.name} ---")
        report_lines.append(f"Method: {results.get('method', 'Unknown')}")
        report_lines.append(f"Timestamp: {results.get('timestamp', 'Unknown')}")

        baseline = results.get("baseline_results", {})
        stabilized = results.get("stabilized_results", {})

        if baseline and stabilized:
            baseline_drift = baseline.get("drift", {}).get("mean_kl", "N/A")
            stabilized_drift = stabilized.get("drift", {}).get("mean_kl", "N/A")
            stabilized_alpha = stabilized.get("alpha", "N/A")

            report_lines.append(f"\nBaseline Results:")
            report_lines.append(f"  Mean Drift (KL): {baseline_drift}")

            report_lines.append(f"\nStabilized Results:")
            report_lines.append(f"  Optimal Alpha: {stabilized_alpha}")
            report_lines.append(f"  Mean Drift (KL): {stabilized_drift}")

            if isinstance(baseline_drift, float) and isinstance(stabilized_drift, float):
                if baseline_drift > 0:
                    reduction = (baseline_drift - stabilized_drift) / baseline_drift * 100
                    report_lines.append(f"  Drift Reduction: {reduction:.1f}%")

    report = "\n".join(report_lines)

    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to {output_path}")
    return report
