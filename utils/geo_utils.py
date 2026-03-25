from __future__ import annotations

from typing import Dict, List

import numpy as np

from dataset.material_dataset import ELEMENTS


TRANSITION_METALS = {"Mo", "W", "V", "Nb", "Ti", "Ni", "Co", "Fe"}


def pairwise_distance_stats(positions: np.ndarray) -> Dict[str, float]:
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    non_zero = dists[dists > 1e-6]
    if non_zero.size == 0:
        return {"mean_distance": 0.0, "min_distance": 0.0, "max_distance": 0.0}
    return {
        "mean_distance": float(non_zero.mean()),
        "min_distance": float(non_zero.min()),
        "max_distance": float(non_zero.max()),
    }


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def proxy_material_metrics(material: Dict) -> Dict[str, float]:
    elements = material["elements"]
    positions = np.array(material["positions"], dtype=float)
    stats = pairwise_distance_stats(positions)
    y_disp = float(np.std(positions[:, 1])) if len(positions) else 0.0

    metal_candidates = [e for e in elements if e in TRANSITION_METALS]
    metal = metal_candidates[0] if metal_candidates else elements[0]
    non_metals = [e for e in elements if e != metal]
    if not non_metals:
        non_metals = [metal]

    metal_en = ELEMENTS[metal]["en"]
    nonmetal_mean_en = float(np.mean([ELEMENTS[e]["en"] for e in non_metals]))
    nonmetal_mean_radius = float(np.mean([ELEMENTS[e]["radius"] for e in non_metals]))
    en_gap = abs(metal_en - nonmetal_mean_en)
    te_penalty = 0.16 if "Te" in elements else 0.0
    janus_bonus = 0.10 if len(set(non_metals)) > 1 else 0.0
    packing_penalty = max(0.0, 0.33 - stats["min_distance"])

    delta_g_h = (
        0.85 * (en_gap - 0.72)
        + 0.35 * (nonmetal_mean_radius - 1.02)
        + 0.30 * (y_disp - 0.28)
        - janus_bonus
        + te_penalty
    )
    thermo = _sigmoid(
        2.4
        - 1.5 * abs(nonmetal_mean_radius - 1.02)
        - 0.8 * abs(en_gap - 0.72)
        - 1.1 * packing_penalty
        - 0.5 * abs(y_disp - 0.28)
        - 0.4 * te_penalty
    )
    kinetic = _sigmoid(
        2.1
        - 1.4 * abs(stats["min_distance"] - 0.42)
        - 0.7 * abs(y_disp - 0.28)
        - 0.4 * te_penalty
        + 0.25 * janus_bonus
    )
    synth = _sigmoid(1.8 + 0.7 * thermo + 0.6 * kinetic - 1.0 * abs(delta_g_h) - 0.3 * te_penalty)
    return {
        "delta_g_h": float(delta_g_h),
        "thermo_stability": float(thermo),
        "kinetic_stability": float(kinetic),
        "synthesizability": float(synth),
    }


def evaluate_stability(positions: np.ndarray, predicted_properties: Dict[str, float]) -> Dict[str, float]:
    stats = pairwise_distance_stats(positions)
    packing_penalty = max(0.0, 0.45 - stats["min_distance"])
    thermo = max(0.0, predicted_properties["thermo_stability"] - 0.5 * packing_penalty)
    kinetic = max(0.0, predicted_properties["kinetic_stability"] - 0.4 * packing_penalty)
    return {
        "thermo_score": thermo,
        "kinetic_score": kinetic,
        "packing_penalty": packing_penalty,
    }


def evaluate_her(delta_g_h: float) -> Dict[str, float]:
    return {
        "delta_g_h": float(delta_g_h),
        "alignment_score": float(np.exp(-abs(delta_g_h))),
    }


def summarize_candidates(candidates: List[Dict]) -> Dict[str, float]:
    delta_g = [item["predicted_properties"]["delta_g_h"] for item in candidates]
    stability = [
        0.5
        * (
            item["predicted_properties"]["thermo_stability"]
            + item["predicted_properties"]["kinetic_stability"]
        )
        for item in candidates
    ]
    synth = [item["predicted_properties"]["synthesizability"] for item in candidates]
    success_rate = sum(
        abs(item["predicted_properties"]["delta_g_h"]) < 0.2
        and item["predicted_properties"]["thermo_stability"] > 0.75
        and item["predicted_properties"]["kinetic_stability"] > 0.75
        and item["predicted_properties"]["synthesizability"] > 0.75
        for item in candidates
    ) / max(len(candidates), 1)
    return {
        "avg_delta_g_abs": float(np.mean(np.abs(delta_g))) if delta_g else 0.0,
        "avg_stability": float(np.mean(stability)) if stability else 0.0,
        "avg_synth": float(np.mean(synth)) if synth else 0.0,
        "success_rate": float(success_rate),
    }
