from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DescendantRecord:
    lineage_id: str
    selection_bpb: float
    report_bpb: float
    improvement_vs_parent: float
    improvement_vs_best: float
    compute_spent: float
    generation: int
    mutation_path: str
    result_path: str
    timestamp: str = field(default_factory=utc_now)


@dataclass
class LineageSelectionMetrics:
    best_descendant_selection_loss: float | None = None
    average_descendant_improvement: float = 0.0
    recent_improvement_slope: float = 0.0
    novelty_bonus: float = 0.0
    compute_penalty: float = 0.0
    lineage_score: float | None = None


@dataclass
class LineageRecord:
    lineage_id: str
    parent_ids: list[str]
    git_commit_hash: str
    checkpoint_path: str | None
    optimizer_state_path: str | None
    recipe_path: str
    recipe_hash: str
    mutation_log_path: str | None
    generation: int
    cumulative_compute_spent: float
    descendant_history: list[DescendantRecord] = field(default_factory=list)
    selection_metrics: LineageSelectionMetrics = field(default_factory=LineageSelectionMetrics)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LineageRecord":
        payload = dict(payload)
        payload["descendant_history"] = [DescendantRecord(**item) for item in payload.get("descendant_history", [])]
        payload["selection_metrics"] = LineageSelectionMetrics(**payload.get("selection_metrics", {}))
        return cls(**payload)
