from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lineage import LineageRecord, utc_now


class LineageStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.lineages_dir = self.root / "lineages"
        self.checkpoints_dir = self.root / "artifacts" / "checkpoints"
        self.optimizer_dir = self.root / "artifacts" / "optimizer"
        self.results_dir = self.root / "artifacts" / "results"
        self.mutations_dir = self.root / "artifacts" / "mutations"
        self.recipes_dir = self.root / "artifacts" / "recipes"
        self.logs_dir = self.root / "artifacts" / "logs"
        for directory in [self.lineages_dir, self.checkpoints_dir, self.optimizer_dir, self.results_dir, self.mutations_dir, self.recipes_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def lineage_path(self, lineage_id: str) -> Path:
        return self.lineages_dir / f"{lineage_id}.json"

    def result_path(self, lineage_id: str) -> Path:
        return self.results_dir / f"{lineage_id}.json"

    def mutation_path(self, lineage_id: str) -> Path:
        return self.mutations_dir / f"{lineage_id}.json"

    def recipe_path(self, lineage_id: str) -> Path:
        return self.recipes_dir / f"{lineage_id}.json"

    def checkpoint_path(self, lineage_id: str) -> Path:
        return self.checkpoints_dir / f"{lineage_id}.pt"

    def optimizer_state_path(self, lineage_id: str) -> Path:
        return self.optimizer_dir / f"{lineage_id}.pt"

    def append_jsonl(self, name: str, row: dict[str, Any]) -> None:
        path = self.logs_dir / name
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def save_lineage(self, lineage: LineageRecord) -> None:
        lineage.updated_at = utc_now()
        self.lineage_path(lineage.lineage_id).write_text(json.dumps(lineage.to_dict(), indent=2) + "\n")

    def load_lineage(self, lineage_id: str) -> LineageRecord:
        return LineageRecord.from_dict(json.loads(self.lineage_path(lineage_id).read_text()))

    def has_lineage(self, lineage_id: str) -> bool:
        return self.lineage_path(lineage_id).exists()

    def list_lineages(self) -> list[LineageRecord]:
        return [LineageRecord.from_dict(json.loads(path.read_text())) for path in sorted(self.lineages_dir.glob("*.json"))]
