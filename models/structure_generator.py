from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch

from models.optimization import MultiObjectiveOptimizer
from utils.geo_utils import proxy_material_metrics


class StructureGenerator:
    def __init__(self, model, dataset, device: torch.device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.optimizer = MultiObjectiveOptimizer()

    def _quantize_node_features(self, node_features: torch.Tensor) -> torch.Tensor:
        library = self.dataset.element_feature_library.to(node_features.device)
        bsz, num_nodes, _ = node_features.shape
        flat = node_features.view(-1, node_features.shape[-1])
        distances = torch.cdist(flat.unsqueeze(0), library.unsqueeze(0)).squeeze(0)
        nearest = distances.argmin(dim=-1)
        quantized = library[nearest].view(bsz, num_nodes, -1)
        return quantized

    def generate(
        self,
        num_candidates: int,
        top_k: int = 10,
        guided: bool = True,
        rerank: bool = True,
    ) -> List[Dict]:
        template_batch = self.dataset.sample_generation_templates(num_candidates)
        node_features = template_batch["node_features"].to(self.device)
        positions = template_batch["positions"].to(self.device)
        mask = template_batch["mask"].to(self.device)
        condition = template_batch["condition"].to(self.device)

        guidance_scale = 1.8 if guided else 0.0
        sampled_features, sampled_positions = self.model.sample(
            node_features=node_features,
            positions=positions,
            mask=mask,
            condition=condition,
            guidance_scale=guidance_scale,
        )
        sampled_features = self._quantize_node_features(sampled_features)
        sampled_positions = torch.tanh(sampled_positions)

        materials = []
        proxy_rows = []
        for idx in range(num_candidates):
            material = self.dataset.decode_material(
                sampled_features[idx].cpu(),
                sampled_positions[idx].cpu(),
                mask[idx].cpu(),
            )
            metrics = proxy_material_metrics(material)
            material["predicted_properties"] = metrics
            materials.append(material)
            proxy_rows.append([
                metrics["delta_g_h"],
                metrics["thermo_stability"],
                metrics["kinetic_stability"],
                metrics["synthesizability"],
            ])

        proxy_tensor = torch.tensor(proxy_rows, dtype=torch.float32)
        score_tensor = self.optimizer.score(proxy_tensor, sampled_features.cpu())
        if rerank:
            selected_indices = torch.topk(score_tensor, k=min(top_k, num_candidates)).indices.tolist()
        else:
            selected_indices = list(range(min(top_k, num_candidates)))

        results = []
        for rank, idx in enumerate(selected_indices, start=1):
            material = materials[idx]
            material["rank"] = rank
            material["guided"] = guided
            material["score_breakdown"] = self.optimizer.summarize(
                proxy_tensor, sampled_features.cpu(), idx
            ).__dict__
            results.append(material)
        return results

    def save_structures(self, structures: List[Dict], output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for material in structures:
            file_path = output_path / f"candidate_{material['rank']:02d}.json"
            with file_path.open("w", encoding="utf-8") as fp:
                json.dump(material, fp, indent=2)
