from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class CandidateScore:
    total: float
    delta_g_alignment: float
    thermo: float
    kinetic: float
    synthesis: float
    novelty: float


class MultiObjectiveOptimizer:
    def __init__(self, weights: Dict[str, float] | None = None):
        self.weights = weights or {
            "delta_g": 0.4,
            "thermo": 0.2,
            "kinetic": 0.2,
            "synthesis": 0.15,
            "novelty": 0.05,
        }

    def _alignment(self, delta_g: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.abs(delta_g))

    def _novelty(self, node_features: torch.Tensor) -> torch.Tensor:
        mean_feat = node_features.mean(dim=(1, 2))
        return torch.sigmoid((mean_feat - mean_feat.mean()) / (mean_feat.std() + 1e-6))

    def score(self, predicted_properties: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        delta_g = predicted_properties[:, 0]
        thermo = predicted_properties[:, 1].sigmoid()
        kinetic = predicted_properties[:, 2].sigmoid()
        synth = predicted_properties[:, 3].sigmoid()
        novelty = self._novelty(node_features)
        score = (
            self.weights["delta_g"] * self._alignment(delta_g)
            + self.weights["thermo"] * thermo
            + self.weights["kinetic"] * kinetic
            + self.weights["synthesis"] * synth
            + self.weights["novelty"] * novelty
        )
        return score

    def summarize(self, predicted_properties: torch.Tensor, node_features: torch.Tensor, index: int) -> CandidateScore:
        score = self.score(predicted_properties, node_features)[index]
        delta_g = predicted_properties[index, 0]
        thermo = predicted_properties[index, 1].sigmoid()
        kinetic = predicted_properties[index, 2].sigmoid()
        synth = predicted_properties[index, 3].sigmoid()
        novelty = self._novelty(node_features)[index]
        return CandidateScore(
            total=float(score.detach().cpu()),
            delta_g_alignment=float(torch.exp(-torch.abs(delta_g)).detach().cpu()),
            thermo=float(thermo.detach().cpu()),
            kinetic=float(kinetic.detach().cpu()),
            synthesis=float(synth.detach().cpu()),
            novelty=float(novelty.detach().cpu()),
        )

