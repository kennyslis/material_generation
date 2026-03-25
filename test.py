import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from dataset.material_dataset import MaterialDataset
from models.diffusion_model import CrystalDiffusionModel
from models.structure_generator import StructureGenerator
from utils.geo_utils import summarize_candidates
from utils.vis import plot_generated_structures, plot_her_distribution, plot_stability_curve


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = CrystalDiffusionModel(
        node_dim=4,
        condition_dim=4,
        hidden_dim=config["hidden_dim"],
        diffusion_steps=config["diffusion_steps"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint_path = Path(args.output_dir) / "checkpoints" / "best_model.pt"
    model, config = load_model(checkpoint_path, device)

    test_dataset = MaterialDataset(Path(args.output_dir) / "dataset", split="test", num_samples=args.test_samples)
    generator = StructureGenerator(model=model, dataset=test_dataset, device=device)

    guided_candidates = generator.generate(num_candidates=args.num_candidates, top_k=args.top_k, guided=True, rerank=True)
    baseline_candidates = generator.generate(num_candidates=args.num_candidates, top_k=args.top_k, guided=False, rerank=False)
    generator.save_structures(guided_candidates, Path(args.output_dir) / "generated_structures")

    guided_summary = summarize_candidates(guided_candidates)
    baseline_summary = summarize_candidates(baseline_candidates)
    comparison = {
        "baseline": baseline_summary,
        "ours": guided_summary,
        "num_saved_structures": len(guided_candidates),
    }

    results_dir = Path(args.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_her_distribution(guided_candidates, results_dir / "her_performance.png")
    plot_stability_curve(guided_candidates, results_dir / "stability_curve.png")
    plot_generated_structures(guided_candidates, results_dir / "generated_structures.png")
    with (results_dir / "comparison_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(comparison, fp, indent=2)

    print(json.dumps(comparison, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Generate and evaluate 2D materials")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--num-candidates", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--test-samples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
