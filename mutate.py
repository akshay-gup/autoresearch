from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any


def _bounded(value: float, factor: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value * factor))


def mutate_recipe(recipe: dict[str, Any], *, rng: random.Random, allow_code_patch: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    child = deepcopy(recipe)
    log: list[dict[str, Any]] = []
    ops = [
        ("embedding_lr", lambda x: _bounded(x, rng.choice([0.8, 0.9, 1.1, 1.25]), 1e-4, 2.0)),
        ("unembedding_lr", lambda x: _bounded(x, rng.choice([0.8, 0.9, 1.1, 1.25]), 1e-5, 0.1)),
        ("matrix_lr", lambda x: _bounded(x, rng.choice([0.8, 0.9, 1.1, 1.25]), 1e-4, 1.0)),
        ("weight_decay", lambda x: _bounded(x, rng.choice([0.5, 0.8, 1.2, 1.5]), 0.0, 1.0)),
        ("warmup_ratio", lambda x: min(0.5, max(0.0, x + rng.choice([-0.02, 0.02, 0.05])))),
        ("device_batch_size", lambda x: int(max(16, min(512, rng.choice([x // 2 if x > 16 else x, x, x * 2]))))),
        ("total_batch_size", lambda x: int(max(2**14, min(2**21, rng.choice([x // 2, x, x * 2]))))),
        ("window_pattern", lambda x: rng.choice(["SSSL", "SLSL", "LLSL", "LLLL"])),
        ("depth", lambda x: int(max(4, min(16, x + rng.choice([-1, 0, 1]))))),
    ]
    for key, fn in rng.sample(ops, k=min(3, len(ops))):
        before = child[key]
        after = fn(before)
        if key == "total_batch_size":
            after = 2 ** round((after).bit_length() - 1)
        child[key] = after
        log.append({"kind": "recipe_mutation", "field": key, "before": before, "after": after})
    child["mutation_seed"] = rng.randint(0, 2**31 - 1)
    if allow_code_patch and rng.random() < 0.1:
        log.append({"kind": "code_patch", "status": "skipped", "reason": "safe automatic code recombination not implemented"})
    return child, log


def recombine_recipe(primary: dict[str, Any], secondary: dict[str, Any], *, rng: random.Random) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    child = deepcopy(primary)
    log = []
    for field in rng.sample(["warmup_ratio", "weight_decay", "window_pattern", "depth"], k=2):
        child[field] = secondary[field]
        log.append({"kind": "recombination", "field": field, "source": "secondary"})
    return child, log


def recipe_hash(recipe: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
