from __future__ import annotations

import gzip
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


ELEMENTS: Dict[str, Dict[str, float]] = {
    "H": {"z": 1, "en": 2.20, "radius": 0.53, "group": 1},
    "B": {"z": 5, "en": 2.04, "radius": 0.84, "group": 13},
    "C": {"z": 6, "en": 2.55, "radius": 0.76, "group": 14},
    "N": {"z": 7, "en": 3.04, "radius": 0.71, "group": 15},
    "O": {"z": 8, "en": 3.44, "radius": 0.66, "group": 16},
    "F": {"z": 9, "en": 3.98, "radius": 0.57, "group": 17},
    "P": {"z": 15, "en": 2.19, "radius": 1.07, "group": 15},
    "S": {"z": 16, "en": 2.58, "radius": 1.05, "group": 16},
    "Cl": {"z": 17, "en": 3.16, "radius": 1.02, "group": 17},
    "Se": {"z": 34, "en": 2.55, "radius": 1.20, "group": 16},
    "Br": {"z": 35, "en": 2.96, "radius": 1.20, "group": 17},
    "Te": {"z": 52, "en": 2.10, "radius": 1.38, "group": 16},
    "I": {"z": 53, "en": 2.66, "radius": 1.39, "group": 17},
    "Ti": {"z": 22, "en": 1.54, "radius": 1.47, "group": 4},
    "V": {"z": 23, "en": 1.63, "radius": 1.53, "group": 5},
    "Cr": {"z": 24, "en": 1.66, "radius": 1.39, "group": 6},
    "Mn": {"z": 25, "en": 1.55, "radius": 1.39, "group": 7},
    "Fe": {"z": 26, "en": 1.83, "radius": 1.26, "group": 8},
    "Co": {"z": 27, "en": 1.88, "radius": 1.25, "group": 9},
    "Ni": {"z": 28, "en": 1.91, "radius": 1.24, "group": 10},
    "Cu": {"z": 29, "en": 1.90, "radius": 1.32, "group": 11},
    "Zn": {"z": 30, "en": 1.65, "radius": 1.22, "group": 12},
    "Ga": {"z": 31, "en": 1.81, "radius": 1.22, "group": 13},
    "Ge": {"z": 32, "en": 2.01, "radius": 1.20, "group": 14},
    "As": {"z": 33, "en": 2.18, "radius": 1.19, "group": 15},
    "Nb": {"z": 41, "en": 1.60, "radius": 1.64, "group": 5},
    "Mo": {"z": 42, "en": 2.16, "radius": 1.39, "group": 6},
    "Tc": {"z": 43, "en": 1.90, "radius": 1.36, "group": 7},
    "Ru": {"z": 44, "en": 2.20, "radius": 1.34, "group": 8},
    "Rh": {"z": 45, "en": 2.28, "radius": 1.34, "group": 9},
    "Pd": {"z": 46, "en": 2.20, "radius": 1.37, "group": 10},
    "Ag": {"z": 47, "en": 1.93, "radius": 1.45, "group": 11},
    "Cd": {"z": 48, "en": 1.69, "radius": 1.44, "group": 12},
    "In": {"z": 49, "en": 1.78, "radius": 1.42, "group": 13},
    "Sn": {"z": 50, "en": 1.96, "radius": 1.39, "group": 14},
    "Sb": {"z": 51, "en": 2.05, "radius": 1.39, "group": 15},
    "W": {"z": 74, "en": 2.36, "radius": 1.41, "group": 6},
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

REAL_DATA_FILENAMES = [
    "2dmatpedia.json",
    "2dmatpedia.json.gz",
    "2dmatpedia_entries.json",
    "2dmatpedia_entries.json.gz",
    "twodmatpedia.json",
    "twodmatpedia.json.gz",
]


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
    item = ELEMENTS.get(symbol)
    if item is None:
        # Fallback encoding for elements not explicitly listed in the lightweight table.
        code = sum(ord(ch) for ch in symbol)
        return np.array(
            [min(code / 300.0, 1.0), 0.50 + (code % 17) / 50.0, 0.45 + (code % 13) / 40.0, 0.35],
            dtype=np.float32,
        )
    return np.array(
        [item["z"] / 100.0, item["en"] / 4.0, item["radius"] / 2.0, item["group"] / 18.0], dtype=np.float32
    )


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def species_to_symbol(site: Dict) -> Optional[str]:
    if isinstance(site.get("label"), str) and site["label"] in ELEMENTS:
        return site["label"]

    species = site.get("species")
    if isinstance(species, list) and species:
        first = species[0]
        if isinstance(first, dict):
            symbol = first.get("element") or first.get("label") or first.get("name")
            if symbol in ELEMENTS:
                return symbol
        elif isinstance(first, str) and first in ELEMENTS:
            return first

    species_string = site.get("species_string") or site.get("element")
    if species_string in ELEMENTS:
        return species_string
    return None


def extract_xy_positions(structure: Dict) -> Optional[np.ndarray]:
    if not isinstance(structure, dict):
        return None

    lattice = structure.get("lattice") or {}
    matrix = lattice.get("matrix") if isinstance(lattice, dict) else None
    sites = structure.get("sites") or []
    positions = []
    for site in sites:
        coords = site.get("xyz")
        if coords is not None and len(coords) >= 2:
            positions.append([safe_float(coords[0]), safe_float(coords[1])])
            continue

        abc = site.get("abc")
        if abc is not None and len(abc) >= 2:
            if matrix is not None and len(matrix) >= 2:
                a_vec = np.array(matrix[0], dtype=np.float32)
                b_vec = np.array(matrix[1], dtype=np.float32)
                frac = np.array([safe_float(abc[0]), safe_float(abc[1])], dtype=np.float32)
                cart = frac[0] * a_vec[:2] + frac[1] * b_vec[:2]
                positions.append(cart.tolist())
            else:
                positions.append([safe_float(abc[0]), safe_float(abc[1])])
    if not positions:
        return None
    return np.array(positions, dtype=np.float32)


def normalize_2d_positions(positions: np.ndarray) -> np.ndarray:
    pos = positions[:, :2].astype(np.float32)
    min_xy = pos.min(axis=0, keepdims=True)
    max_xy = pos.max(axis=0, keepdims=True)
    span = np.maximum(max_xy - min_xy, 1e-6)
    normalized = 2.0 * ((pos - min_xy) / span) - 1.0
    return normalized.astype(np.float32)


def proxy_targets_from_real_fields(entry: Dict, elements: List[str], positions_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    exfoliation_mev = safe_float(
        entry.get("exfoliation_energy", entry.get("exfoliation_energy_per_atom", entry.get("e_exfoliation", 35.0))),
        35.0,
    )
    decomposition_mev = safe_float(
        entry.get("decomposition_energy", entry.get("ehull", entry.get("energy_above_hull", 80.0))),
        80.0,
    )
    band_gap_mev = safe_float(entry.get("bandgap", entry.get("band_gap", 0.0)), 0.0)
    magnetic_moment = abs(safe_float(entry.get("magnetic_moment", entry.get("magmom", 0.0)), 0.0))
    discovery_process = str(entry.get("discovery_process", entry.get("prototype_source", "unknown"))).lower()

    electronegativities = [ELEMENTS[s]["en"] for s in elements if s in ELEMENTS]
    radii = [ELEMENTS[s]["radius"] for s in elements if s in ELEMENTS]
    en_std = float(np.std(electronegativities)) if electronegativities else 0.0
    radius_std = float(np.std(radii)) if radii else 0.0
    y_disp = float(np.std(positions_2d[:, 1])) if len(positions_2d) > 1 else 0.0

    delta_g_h = (
        0.0035 * (exfoliation_mev - 35.0)
        + 0.0020 * decomposition_mev
        - 0.20 * en_std
        + 0.10 * radius_std
        - 0.12 * ("top-down" in discovery_process)
        + 0.06 * min(band_gap_mev / 1000.0, 1.5)
        + 0.03 * min(magnetic_moment, 2.0)
    )

    thermo = sigmoid(2.7 - decomposition_mev / 80.0 - exfoliation_mev / 160.0 - 0.4 * radius_std)
    kinetic = sigmoid(2.4 - abs(y_disp - 0.30) * 2.4 - 0.12 * min(magnetic_moment, 3.0) + 0.25 * (band_gap_mev > 50.0))
    synth = sigmoid(1.9 + 0.9 * thermo + 0.55 * kinetic - 1.1 * abs(delta_g_h) - decomposition_mev / 220.0)

    targets = np.array([delta_g_h, thermo, kinetic, synth], dtype=np.float32)
    condition = np.array([0.0, 0.88, 0.85, 0.82], dtype=np.float32)
    metadata = {
        "material_id": entry.get("material_id") or entry.get("id") or entry.get("uid") or entry.get("_id"),
        "source": "2dmatpedia",
        "discovery_process": entry.get("discovery_process"),
        "exfoliation_energy_mev": exfoliation_mev,
        "decomposition_energy_mev": decomposition_mev,
        "band_gap_mev": band_gap_mev,
    }
    return targets, condition, metadata


class MaterialDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        num_samples: int = 320,
        max_nodes: int = 12,
        seed: int = 7,
        dataset_source: str = "auto",
        real_data_path: str | Path | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_samples = num_samples
        self.max_nodes = max_nodes
        self.seed = seed
        self.dataset_source = dataset_source
        self.real_data_path = Path(real_data_path) if real_data_path else self._discover_real_data_path()
        self.source_name = self._resolve_source_name()
        self.processed_path = self.root_dir / "processed" / f"{self.source_name}_{split}_materials.json"
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)

        if self.processed_path.exists():
            self.records = self._load_records()
        else:
            if self._should_use_real_data():
                self.records = self._build_real_records()
            else:
                self.records = self._build_demo_records()
            self._save_records()

        self.element_symbols = list(ELEMENTS.keys())
        self.element_feature_library = torch.tensor(
            np.stack([element_to_feature(symbol) for symbol in self.element_symbols]), dtype=torch.float32
        )

    def _resolve_source_name(self) -> str:
        if self.dataset_source == "surrogate":
            return "surrogate"
        if self.real_data_path is not None and self.real_data_path.exists():
            return "real2d"
        return "surrogate"

    def _discover_real_data_path(self) -> Optional[Path]:
        raw_dir = self.root_dir / "raw"
        candidates = [raw_dir / name for name in REAL_DATA_FILENAMES]
        env_candidate = None
        env_value = None
        try:
            import os
            env_value = os.environ.get("TWOD_DATA_JSON")
        except Exception:
            env_value = None
        if env_value:
            env_candidate = Path(env_value)
            candidates.insert(0, env_candidate)
        for path in candidates:
            if path.exists():
                return path
        return None

    def _should_use_real_data(self) -> bool:
        if self.dataset_source == "surrogate":
            return False
        if self.dataset_source == "real":
            return self.real_data_path is not None and self.real_data_path.exists()
        return self.real_data_path is not None and self.real_data_path.exists()

    def _split_offset(self) -> int:
        return {"train": 0, "val": 10_000, "test": 20_000}.get(self.split, 0)

    def _read_json_payload(self, path: Path):
        def _load_text_as_json_or_jsonl(content: str):
            content = content.strip()
            if not content:
                return []
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                rows = []
                for line in content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
                return rows

        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as fp:
                return _load_text_as_json_or_jsonl(fp.read())
        with path.open("r", encoding="utf-8") as fp:
            return _load_text_as_json_or_jsonl(fp.read())

    def _build_real_records(self) -> List[MaterialRecord]:
        payload = self._read_json_payload(self.real_data_path)
        if isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                entries = payload["data"]
            elif "results" in payload and isinstance(payload["results"], list):
                entries = payload["results"]
            else:
                entries = list(payload.values())
        else:
            entries = payload

        rng = random.Random(self.seed + self._split_offset())
        parsed_records: List[MaterialRecord] = []
        for raw in entries:
            if not isinstance(raw, dict):
                continue
            structure = raw.get("structure") or raw.get("initial_structure") or raw.get("final_structure")
            positions = extract_xy_positions(structure)
            if positions is None:
                continue
            sites = structure.get("sites") if isinstance(structure, dict) else None
            if not isinstance(sites, list):
                continue
            elements = [species_to_symbol(site) for site in sites]
            if any(symbol is None for symbol in elements):
                continue
            elements = [symbol for symbol in elements if symbol is not None]
            if len(elements) == 0 or len(elements) > self.max_nodes:
                continue
            # Keep records even when some elements are not present in the lightweight lookup table.

            positions_2d = normalize_2d_positions(positions)
            if positions_2d.shape[0] != len(elements):
                continue

            node_features = np.zeros((self.max_nodes, 4), dtype=np.float32)
            node_features[: len(elements)] = np.stack([element_to_feature(symbol) for symbol in elements], axis=0)
            padded_positions = np.zeros((self.max_nodes, 2), dtype=np.float32)
            padded_positions[: len(elements)] = positions_2d
            mask = np.zeros((self.max_nodes,), dtype=np.float32)
            mask[: len(elements)] = 1.0
            targets, condition, metadata = proxy_targets_from_real_fields(raw, elements, positions_2d)
            formula = str(raw.get("formula", raw.get("pretty_formula", ""))) or "".join(elements)
            parsed_records.append(
                MaterialRecord(
                    formula=formula,
                    elements=elements,
                    node_features=node_features,
                    positions=padded_positions,
                    mask=mask,
                    condition=condition,
                    targets=targets,
                    metadata=metadata,
                )
            )

        rng.shuffle(parsed_records)
        if not parsed_records:
            return self._build_demo_records()

        total = len(parsed_records)
        train_cut = max(1, int(total * 0.8))
        val_cut = max(train_cut + 1, int(total * 0.9))
        if self.split == "train":
            selected = parsed_records[:train_cut]
        elif self.split == "val":
            selected = parsed_records[train_cut:val_cut]
        else:
            selected = parsed_records[val_cut:]

        if not selected:
            selected = parsed_records[: min(len(parsed_records), max(8, self.num_samples))]
        if self.num_samples and len(selected) > self.num_samples:
            selected = selected[: self.num_samples]
        return selected

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
            elements = [metal, nonmetal_a, nonmetal_b, metal, nonmetal_a, nonmetal_b]
            formula = f"{metal}{nonmetal_a}{nonmetal_b if nonmetal_b != nonmetal_a else nonmetal_a}"

            prototype = PROTOTYPES[prototype_name].copy()
            jitter = np.random.default_rng(self.seed + self._split_offset() + idx).normal(0, 0.035, size=prototype.shape)
            positions = (prototype + jitter).astype(np.float32)
            positions = np.clip(positions, -0.2, 1.2)
            positions = normalize_2d_positions(positions)

            node_features = np.zeros((self.max_nodes, 4), dtype=np.float32)
            node_features[:6] = np.stack([element_to_feature(symbol) for symbol in elements], axis=0)
            padded_positions = np.zeros((self.max_nodes, 2), dtype=np.float32)
            padded_positions[:6] = positions
            mask = np.zeros((self.max_nodes,), dtype=np.float32)
            mask[:6] = 1.0

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
            metadata = {"prototype": prototype_name, "sample_id": idx, "source": "surrogate"}
            records.append(
                MaterialRecord(
                    formula=formula,
                    elements=elements,
                    node_features=node_features,
                    positions=padded_positions,
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
        formula = "".join(element_names[: min(3, len(element_names))])
        return {
            "formula": formula,
            "elements": element_names,
            "positions": positions[:valid_nodes].tolist(),
        }

