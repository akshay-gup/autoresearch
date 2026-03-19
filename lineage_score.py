from __future__ import annotations

from lineage import DescendantRecord, LineageSelectionMetrics


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    indices = list(range(len(values)))
    mean_x = sum(indices) / len(indices)
    mean_y = sum(values) / len(values)
    denom = sum((x - mean_x) ** 2 for x in indices)
    if denom == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(indices, values)) / denom


def compute_lineage_score(descendants: list[DescendantRecord], novelty_bonus: float = 0.0, compute_penalty_weight: float = 0.01) -> LineageSelectionMetrics:
    if not descendants:
        return LineageSelectionMetrics(lineage_score=float("-inf"))
    best_loss = min(item.selection_bpb for item in descendants)
    improvements = [item.improvement_vs_parent for item in descendants]
    avg_improvement = sum(improvements) / len(improvements)
    slope = -_slope([item.selection_bpb for item in descendants])
    compute_penalty = compute_penalty_weight * sum(item.compute_spent for item in descendants)
    score = -best_loss + avg_improvement + slope + novelty_bonus - compute_penalty
    return LineageSelectionMetrics(
        best_descendant_selection_loss=best_loss,
        average_descendant_improvement=avg_improvement,
        recent_improvement_slope=slope,
        novelty_bonus=novelty_bonus,
        compute_penalty=compute_penalty,
        lineage_score=score,
    )
