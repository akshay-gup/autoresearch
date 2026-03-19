from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from lineage import DescendantRecord, LineageRecord
from lineage_score import compute_lineage_score
from lineage_store import LineageStore
from mutate import mutate_recipe, recipe_hash, recombine_recipe, write_json

ROOT = Path(__file__).resolve().parent
GERMLINE_FILES = ["train.py", "prepare.py", "README.md", "pyproject.toml"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def copy_germline_snapshot(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for rel in GERMLINE_FILES:
        src = ROOT / rel
        if src.exists():
            shutil.copy2(src, target_dir / rel)


def load_recipe(path: Path) -> dict:
    return json.loads(path.read_text())


def init_lineage(args):
    store = LineageStore(args.archive_root)
    if store.has_lineage(args.lineage_id):
        raise SystemExit(f"Lineage {args.lineage_id} already exists")
    recipe_path = store.recipe_path(args.lineage_id)
    if args.recipe:
        recipe_payload = load_recipe(Path(args.recipe))
    else:
        metrics_path = store.results_dir / f"{args.lineage_id}_bootstrap_recipe.json"
        subprocess.run([sys.executable, "train.py", "--export-recipe", str(metrics_path)], cwd=ROOT, check=True)
        recipe_payload = load_recipe(metrics_path)
    write_json(recipe_path, recipe_payload)
    lineage = LineageRecord(
        lineage_id=args.lineage_id,
        parent_ids=[],
        git_commit_hash=git_commit(),
        checkpoint_path=args.checkpoint,
        optimizer_state_path=args.optimizer_state,
        recipe_path=str(recipe_path),
        recipe_hash=recipe_hash(recipe_payload),
        mutation_log_path=None,
        generation=0,
        cumulative_compute_spent=0.0,
        metadata={
            "germline_snapshot_dir": str((Path(args.archive_root) / "artifacts" / "germline" / args.lineage_id)),
            "initialized_at": utc_now(),
        },
    )
    copy_germline_snapshot(Path(lineage.metadata["germline_snapshot_dir"]))
    store.save_lineage(lineage)
    store.append_jsonl("events.jsonl", {"event": "init_lineage", "lineage_id": lineage.lineage_id, "timestamp": utc_now()})
    print(f"Initialized lineage {lineage.lineage_id}")


def run_train(recipe_path: Path, load_checkpoint: str | None, save_checkpoint: Path, result_path: Path, time_budget: int, selection_split: str, report_split: str):
    cmd = [
        sys.executable,
        "train.py",
        "--recipe", str(recipe_path),
        "--save-checkpoint", str(save_checkpoint),
        "--save-optimizer",
        "--metrics-json", str(result_path),
        "--time-budget", str(time_budget),
        "--selection-split", selection_split,
        "--report-split", report_split,
    ]
    if load_checkpoint:
        cmd.extend(["--load-checkpoint", load_checkpoint])
    subprocess.run(cmd, cwd=ROOT, check=True)


def evolve_generation(args):
    store = LineageStore(args.archive_root)
    rng = random.Random(args.seed)
    parent_ids = args.parents or [item.lineage_id for item in store.list_lineages()]
    if not parent_ids:
        raise SystemExit("No parent lineages available")
    scores = []
    for parent_id in parent_ids:
        parent = store.load_lineage(parent_id)
        parent_recipe = load_recipe(Path(parent.recipe_path))
        descendants = []
        parent_baseline = None
        if parent.descendant_history:
            parent_baseline = min(item.selection_bpb for item in parent.descendant_history)
        elif parent.metadata.get("baseline_selection_bpb") is not None:
            parent_baseline = parent.metadata["baseline_selection_bpb"]
        else:
            parent_baseline = float("inf")
        for child_idx in range(args.children_per_parent):
            child_id = f"{parent.lineage_id}.g{parent.generation+1}.c{child_idx:02d}"
            mutation_log = []
            child_recipe, mutation_log = mutate_recipe(parent_recipe, rng=rng)
            parent_ids_for_child = [parent.lineage_id]
            if args.enable_recombination and len(parent_ids) > 1 and rng.random() < args.recombination_rate:
                other_id = rng.choice([pid for pid in parent_ids if pid != parent.lineage_id])
                other = store.load_lineage(other_id)
                other_recipe = load_recipe(Path(other.recipe_path))
                child_recipe, recomb_log = recombine_recipe(child_recipe, other_recipe, rng=rng)
                mutation_log.extend(recomb_log)
                parent_ids_for_child.append(other_id)
            recipe_path = store.recipe_path(child_id)
            mutation_path = store.mutation_path(child_id)
            result_path = store.result_path(child_id)
            checkpoint_path = store.checkpoint_path(child_id)
            write_json(recipe_path, child_recipe)
            write_json(mutation_path, {
                "lineage_id": child_id,
                "parent_ids": parent_ids_for_child,
                "recipe_hash": recipe_hash(child_recipe),
                "mutations": mutation_log,
                "timestamp": utc_now(),
            })
            load_checkpoint_path = parent.checkpoint_path if parent.checkpoint_path and Path(parent.checkpoint_path).exists() else None
            run_train(recipe_path, load_checkpoint_path, checkpoint_path, result_path, args.time_budget, args.selection_split, args.report_split)
            result = json.loads(result_path.read_text())
            selection_bpb = result["selection_bpb"]
            report_bpb = result["report_bpb"]
            improvement_vs_parent = 0.0 if parent_baseline == float("inf") else parent_baseline - selection_bpb
            improvement_vs_best = parent.selection_metrics.best_descendant_selection_loss - selection_bpb if parent.selection_metrics.best_descendant_selection_loss is not None else improvement_vs_parent
            descendants.append(DescendantRecord(
                lineage_id=child_id,
                selection_bpb=selection_bpb,
                report_bpb=report_bpb,
                improvement_vs_parent=improvement_vs_parent,
                improvement_vs_best=improvement_vs_best,
                compute_spent=float(result["training_seconds"]),
                generation=parent.generation + 1,
                mutation_path=str(mutation_path),
                result_path=str(result_path),
            ))
            child_lineage = LineageRecord(
                lineage_id=child_id,
                parent_ids=parent_ids_for_child,
                git_commit_hash=git_commit(),
                checkpoint_path=str(checkpoint_path),
                optimizer_state_path=str(checkpoint_path),
                recipe_path=str(recipe_path),
                recipe_hash=recipe_hash(child_recipe),
                mutation_log_path=str(mutation_path),
                generation=parent.generation + 1,
                cumulative_compute_spent=parent.cumulative_compute_spent + float(result["training_seconds"]),
                descendant_history=[],
                metadata={
                    "selection_bpb": selection_bpb,
                    "report_bpb": report_bpb,
                    "inherited_checkpoint": load_checkpoint_path,
                    "germline_files": GERMLINE_FILES,
                },
            )
            store.save_lineage(child_lineage)
            store.append_jsonl("events.jsonl", {"event": "child_evaluated", "parent_id": parent.lineage_id, "child_id": child_id, "selection_bpb": selection_bpb, "timestamp": utc_now()})
        novelty_bonus = 0.0
        if args.novelty_bonus:
            unique_patterns = len({load_recipe(Path(store.load_lineage(d.lineage_id).recipe_path))["window_pattern"] for d in descendants})
            novelty_bonus = args.novelty_bonus * unique_patterns / max(len(descendants), 1)
        parent.descendant_history.extend(descendants)
        parent.cumulative_compute_spent += sum(item.compute_spent for item in descendants)
        parent.selection_metrics = compute_lineage_score(descendants, novelty_bonus=novelty_bonus, compute_penalty_weight=args.compute_penalty_weight)
        if descendants:
            best_child = min(descendants, key=lambda item: item.selection_bpb)
            parent.metadata["baseline_selection_bpb"] = min(parent_baseline, best_child.selection_bpb) if parent_baseline != float("inf") else best_child.selection_bpb
        store.save_lineage(parent)
        scores.append(parent)
    promoted = sorted(scores, key=lambda lineage: lineage.selection_metrics.lineage_score or float("-inf"), reverse=True)[: args.promote_top_k]
    promotion_summary = {
        "timestamp": utc_now(),
        "promoted": [item.lineage_id for item in promoted],
        "scores": {item.lineage_id: item.selection_metrics.lineage_score for item in scores},
    }
    store.append_jsonl("selection.jsonl", promotion_summary)
    print(json.dumps(promotion_summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lineage-level evolutionary search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a parent lineage")
    init_parser.add_argument("--archive-root", default="archive")
    init_parser.add_argument("--lineage-id", required=True)
    init_parser.add_argument("--recipe", default=None)
    init_parser.add_argument("--checkpoint", default=None)
    init_parser.add_argument("--optimizer-state", default=None)

    evolve_parser = subparsers.add_parser("generation", help="Run one generation of lineage search")
    evolve_parser.add_argument("--archive-root", default="archive")
    evolve_parser.add_argument("--parents", nargs="*", default=None)
    evolve_parser.add_argument("--children-per-parent", type=int, default=int(os.environ.get("AUTORESEARCH_CHILDREN", "4")))
    evolve_parser.add_argument("--time-budget", type=int, default=60)
    evolve_parser.add_argument("--selection-split", default="selection")
    evolve_parser.add_argument("--report-split", default="report")
    evolve_parser.add_argument("--seed", type=int, default=1234)
    evolve_parser.add_argument("--promote-top-k", type=int, default=1)
    evolve_parser.add_argument("--compute-penalty-weight", type=float, default=0.01)
    evolve_parser.add_argument("--novelty-bonus", type=float, default=0.02)
    evolve_parser.add_argument("--enable-recombination", action="store_true")
    evolve_parser.add_argument("--recombination-rate", type=float, default=0.1)

    args = parser.parse_args()
    if args.command == "init":
        init_lineage(args)
    else:
        evolve_generation(args)
