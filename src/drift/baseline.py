"""
Baseline Prompts Manager

Manages a fixed set of prompts unrelated to edited knowledge for measuring drift.
Pre-computes and stores baseline logits for comparison after editing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


# Default set of diverse prompts unrelated to typical model editing facts
DEFAULT_BASELINE_PROMPTS = [
    # Physics
    "The speed of light in a vacuum is approximately",
    "Newton's first law of motion states that",
    "The formula for kinetic energy is",
    "Entropy in thermodynamics measures",

    # Mathematics
    "The Pythagorean theorem states that in a right triangle",
    "The derivative of sin(x) with respect to x is",
    "A prime number is defined as",
    "The value of pi to five decimal places is",

    # History
    "The ancient Egyptian pyramids were built approximately",
    "The Roman Empire fell in the year",
    "The Industrial Revolution began in",
    "World War I started in the year",

    # Geography
    "Mount Everest is located in",
    "The Amazon River flows through",
    "The Sahara Desert is located in",
    "The Pacific Ocean is the",

    # General Knowledge
    "Water molecules consist of",
    "Photosynthesis converts sunlight into",
    "DNA stands for",
    "The human heart has",

    # Language/Literature
    "Shakespeare wrote the play",
    "The protagonist in a story is",
    "An adjective is a word that",
    "The alphabet has",

    # Science
    "The periodic table organizes elements by",
    "Gravity causes objects to",
    "Sound travels as waves through",
    "The Earth orbits the Sun in approximately",
]


@dataclass
class BaselinePromptsManager:
    """
    Manages baseline prompts and their pre-computed logits for drift measurement.

    Attributes:
        prompts: List of baseline prompts
        baseline_logits: Dict mapping prompts to their original logits
        baseline_log_probs: Dict mapping prompts to target log probabilities
        model_name: Name of the model used for baseline computation
    """
    prompts: List[str] = field(default_factory=list)
    baseline_logits: Dict[str, torch.Tensor] = field(default_factory=dict)
    baseline_log_probs: Dict[str, float] = field(default_factory=dict)
    model_name: str = ""

    def __post_init__(self):
        if not self.prompts:
            self.prompts = DEFAULT_BASELINE_PROMPTS.copy()

    def compute_baselines(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda"
    ) -> None:
        """
        Compute and store baseline logits for all prompts using the unedited model.

        Args:
            model: The unedited model
            tokenizer: The tokenizer
            device: Device to use for computation
        """
        print(f"Computing baselines for {len(self.prompts)} prompts...")
        model.eval()

        self.model_name = model.config._name_or_path
        self.baseline_logits = {}
        self.baseline_log_probs = {}

        with torch.no_grad():
            for i, prompt in enumerate(self.prompts):
                # Tokenize
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True
                ).to(device)

                # Get model output
                outputs = model(**inputs)

                # Store last-token logits
                last_logits = outputs.logits[0, -1, :].cpu()
                self.baseline_logits[prompt] = last_logits

                # Store top-1 log probability for reference
                log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)
                top_log_prob = log_probs.max().item()
                self.baseline_log_probs[prompt] = top_log_prob

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(self.prompts)} prompts")

        print(f"Baselines computed for {len(self.baseline_logits)} prompts")

    def save(self, path: str) -> None:
        """
        Save baseline data to disk.

        Args:
            path: Path to save directory
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save prompts and metadata
        metadata = {
            "prompts": self.prompts,
            "model_name": self.model_name,
            "num_prompts": len(self.prompts),
            "log_probs": self.baseline_log_probs
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save logits as tensor file
        if self.baseline_logits:
            logits_data = {
                prompt: logits for prompt, logits in self.baseline_logits.items()
            }
            torch.save(logits_data, save_dir / "baseline_logits.pt")

        print(f"Baselines saved to {save_dir}")

    @classmethod
    def load(cls, path: str) -> 'BaselinePromptsManager':
        """
        Load baseline data from disk.

        Args:
            path: Path to save directory

        Returns:
            BaselinePromptsManager with loaded data
        """
        load_dir = Path(path)

        # Load metadata
        with open(load_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Load logits
        logits_path = load_dir / "baseline_logits.pt"
        if logits_path.exists():
            baseline_logits = torch.load(logits_path)
        else:
            baseline_logits = {}

        manager = cls(
            prompts=metadata["prompts"],
            baseline_logits=baseline_logits,
            baseline_log_probs=metadata.get("log_probs", {}),
            model_name=metadata.get("model_name", "")
        )

        print(f"Loaded baselines from {load_dir} ({len(manager.prompts)} prompts)")
        return manager

    def add_prompt(self, prompt: str) -> None:
        """Add a new prompt to the baseline set."""
        if prompt not in self.prompts:
            self.prompts.append(prompt)
            # Mark that baseline needs recomputation
            if prompt in self.baseline_logits:
                del self.baseline_logits[prompt]

    def get_baseline_logits(self, prompt: str) -> Optional[torch.Tensor]:
        """Get baseline logits for a specific prompt."""
        return self.baseline_logits.get(prompt)

    def has_baselines(self) -> bool:
        """Check if baselines have been computed."""
        return len(self.baseline_logits) > 0

    def __repr__(self):
        return (f"BaselinePromptsManager("
                f"num_prompts={len(self.prompts)}, "
                f"baselines_computed={self.has_baselines()}, "
                f"model={self.model_name})")


def create_custom_baseline_prompts(
    domain: str = "general",
    num_prompts: int = 20
) -> List[str]:
    """
    Create a custom set of baseline prompts for a specific domain.

    Args:
        domain: Domain for prompts ('general', 'science', 'history', etc.)
        num_prompts: Number of prompts to create

    Returns:
        List of prompts
    """
    domain_prompts = {
        "science": [
            "The chemical formula for water is",
            "Cells are the basic unit of",
            "The nucleus of an atom contains",
            "Evolution is driven by natural",
            "The mitochondria is known as the",
            "Electrons have a negative",
            "The Big Bang theory explains",
            "Photons are particles of",
            "Carbon dioxide is composed of",
            "The human brain contains approximately",
        ],
        "history": [
            "The Declaration of Independence was signed in",
            "The French Revolution began in",
            "Ancient Greece is known for",
            "The Renaissance started in",
            "The Cold War lasted from",
            "The printing press was invented by",
            "The Great Wall of China was built",
            "The Byzantine Empire lasted until",
            "The Magna Carta was signed in",
            "Ancient Rome was founded in",
        ],
        "math": [
            "The square root of 144 is",
            "The sum of angles in a triangle is",
            "Euler's number e is approximately",
            "The factorial of 5 equals",
            "A polygon with eight sides is called",
            "The circumference of a circle is",
            "The quadratic formula solves equations of",
            "Logarithm base 10 of 100 equals",
            "The golden ratio is approximately",
            "Integration is the reverse of",
        ],
    }

    if domain in domain_prompts:
        prompts = domain_prompts[domain][:num_prompts]
    else:
        prompts = DEFAULT_BASELINE_PROMPTS[:num_prompts]

    return prompts
