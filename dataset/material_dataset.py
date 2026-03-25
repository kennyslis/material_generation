from __future__ import annotations
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


ELEMENTS: Dict[str, Dict[str, float]] = {
    "Mo": {"z": 42, "en": 2.16, "radius": 1.39, "group": 6},
    "W": {"z": 74, "en": 2.36, "radius": 1.41, "group": 6},
    "V": {"z": 23, "en": 1.63, "radius": 1.53, "group": 5},
    "Nb": {"z": 41, "en": 1.60, "radius": 1.64, "group": 5},
    "Ti": {"z": 22, "en": 1.54, "radius": 1.47, "group": 4},
    "Ni": {"z": 28, "en": 1.91, "radius": 1.24, "group": 10},
    "Co": {"z": 27, "en": 1.88, "radius": 1.25, "group": 9},
    "Fe": {"z": 26, "en": 1.83, "radius": 1.26, "group": 8},
    "S": {"z": 16, "en": 2.58, "radius": 1.05, "group": 16},
    "Se": {"z": 34, "en": 2.55, "radius": 1.20, "group": 16},
    "Te": {"z": 52, "en": 2.10, "radius": 1.38, "group": 16},
    "N": {"z": 7, "en": 3.04, "radius": 0.71, "group": 15},
    "P": {"z": 15, "en": 2.19, "radius": 1.07, "group": 15},
    "C": {"z": 6, "en": 2.55, "radius": 0.76, "group": 14},
    "B": {"z": 5, "en": 2.04, "radius": 0.84, "group": 13},
}

PROTOTYPES = {
    "hex": np.array(
        [[0.0, 0.0], [0.5, 0.2887], [0.0, 0.5774], [0.5, 0.8660], [1.0, 0.5774], [1.0, 0.0]],
        dtype=np.float32,
    ),
    "rect": np.array(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0, 0.5]],
        dtype=np.float32,
    ),
    "janus": np.array(
        [[0.0, 0.1], [0.35, 0.3], [0.7, 0.1], [0.0, 0.8], [0.35, 0.6], [0.7, 0.8]],
        dtype=np.float32,
    ),
}


@dataclass
class MaterialRecord:
    formula: str
    elements: List[str]
    node_features: np.ndarray
    positions: np.ndarray
    mask: np.ndarray
    condition: np.ndarray
    targets: np.ndarray
    metadata: Dict


def element_to_feature(symbol: str) -> np.ndarray:
    item = ELEMENTS[symbol]
    return np.array(
        [item["z"] / 100.0, item["en"] / 4.0, item["radius"] / 2.0, item["group"] / 18.0], dtype=np.float32
    )


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class MaterialDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        num_samples: int = 320,
        max_nodes: int = 6,
        seed: int = 7,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_samples = num_samples
        self.max_nodes = max_nodes
        self.seed = seed
        self.processed_path = self.root_dir / "processed" / f"{split}_materials.json"
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)

        if self.processed_path.exists():
            self.records = self._load_records()
        else:
            self.records = self._build_demo_records()
            self._save_records()

        self.element_symbols = list(ELEMENTS.keys())
        self.element_feature_library = torch.tensor(
            np.stack([element_to_feature(symbol) for symbol in self.element_symbols]), dtype=torch.float32
        )

    def _split_offset(self) -> int:
        return {"train": 0, "val": 10_000, "test": 20_000}.get(self.split, 0)

    def _build_demo_records(self) -> List[MaterialRecord]:
        rng = random.Random(self.seed + self._split_offset())
        records = []
        transition_metals = ["Mo", "W", "V", "Nb", "Ti", "Ni", "Co", "Fe"]
        non_metals = ["S", "Se", "Te", "N", "P", "C", "B"]
        prototypes = list(PROTOTYPES.keys())

        for idx in range(self.num_samples):
            metal = rng.choice(transition_metals)
            nonmetal_a = rng.choice(non_metals)
            nonmetal_b = nonmetal_a if rng.random() < 0.7 else rng.choice(non_metals)
            prototype_name = rng.choice(prototypes)
            formula = f"{metal}{nonmetal_a}{nonmetal_b if nonmetal_b != nonmetal_a else nonmetal_a}"
            elements = [metal, nonmetal_a, nonmetal_b, metal, nonmetal_a, nonmetal_b]

            prototype = PROTOTYPES[prototype_name].copy()
            jitter = np.random.default_rng(self.seed + self._split_offset() + idx).normal(0, 0.035, size=prototype.shape)
            positions = (prototype + jitter).astype(np.float32)
            positions = np.clip(positions, -0.2, 1.2)

            node_features = np.stack([element_to_feature(symbol) for symbol in elements], axis=0)
            mask = np.ones((self.max_nodes,), dtype=np.float32)

            metal_data = ELEMENTS[metal]
            nonmetal_mean_en = (ELEMENTS[nonmetal_a]["en"] + ELEMENTS[nonmetal_b]["en"]) / 2.0
            en_gap = abs(metal_data["en"] - nonmetal_mean_en)
            radius_mean = (ELEMENTS[nonmetal_a]["radius"] + ELEMENTS[nonmetal_b]["radius"]) / 2.0
            position_dispersion = float(np.std(positions[:, 1]))
            janus_bonus = 0.12 if nonmetal_a != nonmetal_b else 0.0
            hex_bonus = 0.08 if prototype_name == "hex" else 0.0
            te_penalty = 0.18 if "Te" in (nonmetal_a, nonmetal_b) else 0.0

            delta_g_h = (
                0.9 * (en_gap - 0.65)
                + 0.45 * (radius_mean - 1.05)
                - janus_bonus
                - hex_bonus
                + te_penalty
                + rng.uniform(-0.12, 0.12)
            )
            thermo_stability = sigmoid(
                2.6
                - 1.8 * abs(radius_mean - 1.05)
                - 0.9 * abs(en_gap - 0.75)
                + hex_bonus
                - 0.6 * position_dispersion
                + rng.uniform(-0.25, 0.25)
            )
            kinetic_stability = sigmoid(
                2.2
                - 1.4 * position_dispersion
                + 0.4 * hex_bonus
                + 0.2 * janus_bonus
                - 0.5 * te_penalty
                + rng.uniform(-0.25, 0.25)
            )
            synth = sigmoid(
                1.9
                + 0.7 * thermo_stability
                + 0.5 * kinetic_stability
                - 0.7 * abs(delta_g_h)
                - 0.35 * te_penalty
                + rng.uniform(-0.3, 0.3)
            )

            targets = np.array([delta_g_h, thermo_stability, kinetic_stability, synth], dtype=np.float32)
            condition = np.array([0.0, 0.88, 0.85, 0.82], dtype=np.float32)
            metadata = {"prototype": prototype_name, "sample_id": idx}
            records.append(
                MaterialRecord(
                    formula=formula,
                    elements=elements,
                    node_features=node_features,
                    positions=positions,
                    mask=mask,
                    condition=condition,
                    targets=targets,
                    metadata=metadata,
                )
            )
        return records

    def _save_records(self) -> None:
        payload = []
        for record in self.records:
            payload.append(
                {
                    "formula": record.formula,
                    "elements": record.elements,
                    "node_features": record.node_features.tolist(),
                    "positions": record.positions.tolist(),
                    "mask": record.mask.tolist(),
                    "condition": record.condition.tolist(),
                    "targets": record.targets.tolist(),
                    "metadata": record.metadata,
                }
            )
        with self.processed_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    def _load_records(self) -> List[MaterialRecord]:
        with self.processed_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        records = []
        for item in payload:
            records.append(
                MaterialRecord(
                    formula=item["formula"],
                    elements=item["elements"],
                    node_features=np.array(item["node_features"], dtype=np.float32),
                    positions=np.array(item["positions"], dtype=np.float32),
                    mask=np.array(item["mask"], dtype=np.float32),
                    condition=np.array(item["condition"], dtype=np.float32),
                    targets=np.array(item["targets"], dtype=np.float32),
                    metadata=item["metadata"],
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        return {
            "node_features": torch.tensor(record.node_features, dtype=torch.float32),
            "positions": torch.tensor(record.positions, dtype=torch.float32),
            "mask": torch.tensor(record.mask, dtype=torch.float32),
            "condition": torch.tensor(record.condition, dtype=torch.float32),
            "targets": torch.tensor(record.targets, dtype=torch.float32),
        }

    def sample_generation_templates(self, num_candidates: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + 99 + self._split_offset())
        indices = rng.choice(len(self.records), size=num_candidates, replace=True)
        batch = [self.records[int(idx)] for idx in indices]
        return {
            "node_features": torch.tensor(np.stack([item.node_features for item in batch]), dtype=torch.float32),
            "positions": torch.tensor(np.stack([item.positions for item in batch]), dtype=torch.float32),
            "mask": torch.tensor(np.stack([item.mask for item in batch]), dtype=torch.float32),
            "condition": torch.tensor(np.stack([item.condition for item in batch]), dtype=torch.float32),
        }

    def decode_material(self, node_features: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> Dict:
        valid_nodes = int(mask.sum().item())
        element_names = []
        for idx in range(valid_nodes):
            feature = node_features[idx].unsqueeze(0)
            distances = torch.cdist(feature, self.element_feature_library).squeeze(0)
            element_names.append(self.element_symbols[int(distances.argmin().item())])
        formula = "".join(element_names[:3])
        return {
            "formula": formula,
            "elements": element_names,
            "positions": positions[:valid_nodes].tolist(),
        }

