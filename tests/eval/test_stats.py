from __future__ import annotations

from docsqa.eval.stats import paired_bootstrap_ci, spearman, verdict


def test_bootstrap_ci_excludes_zero_for_clear_gap_and_spans_zero_for_noise() -> None:
    mean_diff, lower, upper = paired_bootstrap_ci(
        [1.0] * 40, [0.0] * 40, n_boot=2_000, seed=7
    )
    assert mean_diff == 1.0
    assert lower > 0.0
    assert upper > 0.0

    _, lower, upper = paired_bootstrap_ci(
        [0.0, 1.0] * 20, [1.0, 0.0] * 20, n_boot=2_000, seed=7
    )
    assert lower <= 0 <= upper


def test_verdict_labels() -> None:
    assert verdict(0.01, 0.4) == "better"
    assert verdict(-0.4, -0.01) == "worse"
    assert verdict(-0.1, 0.2) == "no_detectable_difference"
    assert spearman([1, 2, 3], [2, 4, 6]) == 1.0
