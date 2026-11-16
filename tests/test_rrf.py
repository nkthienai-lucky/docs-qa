from __future__ import annotations

from docsqa.bm25 import reciprocal_rank_fusion


def test_rrf_matches_hand_computed_scores() -> None:
    # Ranking A: [a, b, c]; ranking B: [c, b]; k = 60.
    #   a: 1/61                  (rank 1 in A only)
    #   b: 1/62 + 1/62 = 2/62    (rank 2 in both)
    #   c: 1/63 + 1/61           (rank 3 in A, rank 1 in B)
    # c > b > a, no ties.
    a = 1 / 61
    b = 1 / 62 + 1 / 62
    c = 1 / 63 + 1 / 61

    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b"]], k=60)

    assert fused == [("c", c), ("b", b), ("a", a)]


def test_rrf_uses_configurable_k() -> None:
    # Same rankings with k = 1:
    #   a: 1/2
    #   b: 1/3 + 1/3 = 2/3
    #   c: 1/4 + 1/2 = 0.75
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b"]], k=1)

    assert fused == [
        ("c", 1 / 2 + 1 / 4),
        ("b", 1 / 3 + 1 / 3),
        ("a", 1 / 2),
    ]


def test_rrf_prefers_docs_in_both_lists() -> None:
    # "both" appears at rank 3 in each list; "only_a" and "only_b" are rank 1
    # in a single list each. 2/(60+3) > 1/(60+1), so the doc present in both
    # rankings must fuse above either single-list rank-1 doc.
    fused = reciprocal_rank_fusion(
        [["only_a", "x", "both"], ["only_b", "y", "both"]], k=60
    )

    ranked = [doc for doc, _ in fused]
    assert ranked.index("both") < ranked.index("only_a")
    assert ranked.index("both") < ranked.index("only_b")


def test_rrf_single_ranking_preserves_order() -> None:
    fused = reciprocal_rank_fusion([["x", "y", "z"]], k=60)

    assert [doc for doc, _ in fused] == ["x", "y", "z"]
    assert fused[0][1] == 1 / 61
