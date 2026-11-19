from __future__ import annotations

import random
from statistics import mean
from typing import Literal

Verdict = Literal["better", "worse", "no_detectable_difference"]


def paired_bootstrap_ci(
    a: list[float],
    b: list[float],
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and have equal lengths")
    if n_boot <= 0 or not 0 < alpha < 1:
        raise ValueError("n_boot must be positive and alpha must be between 0 and 1")
    differences = [left - right for left, right in zip(a, b)]
    rng = random.Random(seed)
    samples = sorted(
        mean(rng.choices(differences, k=len(differences))) for _ in range(n_boot)
    )
    lo_index = max(0, int((alpha / 2) * n_boot))
    hi_index = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return mean(differences), samples[lo_index], samples[hi_index]


def verdict(lo: float, hi: float) -> Verdict:
    if lo > 0:
        return "better"
    if hi < 0:
        return "worse"
    return "no_detectable_difference"


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for position in order[start:end]:
            ranks[position] = average_rank
        start = end
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Spearman inputs must have equal lengths of at least two")
    ranks_a = _rank(a)
    ranks_b = _rank(b)
    mean_a = mean(ranks_a)
    mean_b = mean(ranks_b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(ranks_a, ranks_b))
    variance_a = sum((x - mean_a) ** 2 for x in ranks_a)
    variance_b = sum((y - mean_b) ** 2 for y in ranks_b)
    if variance_a == 0 or variance_b == 0:
        return 0.0
    return covariance / (variance_a * variance_b) ** 0.5
