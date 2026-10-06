"""APPS problems (Control Arena version, with verified reference solutions)."""

import hashlib
import random
from dataclasses import dataclass

from probe_evasion import config


@dataclass
class Problem:
    problem_id: str
    question: str
    inputs: list[str]
    outputs: list[str]
    reference: str  # a reference solution that the dataset marks as passing its tests

    @property
    def tests(self) -> list[tuple[str, str]]:
        return list(zip(self.inputs, self.outputs))[: config.MAX_TESTS_PER_PROBLEM]


def split_of(problem_id: str) -> str:
    h = hashlib.sha256(f"{config.SPLIT_SEED}:{problem_id}".encode()).digest()
    return "test" if h[0] / 256 < config.TEST_FRACTION else "train"


def load_problems(split: str, difficulty: str = "interview") -> list[Problem]:
    from datasets import load_dataset

    ds = load_dataset(config.APPS_DATASET, split="train")
    problems = []
    for row in ds:
        if row["difficulty"] != difficulty or not row["solution_passes_tests"] or row["is_nondeterministic"]:
            continue
        if not row["inputs"] or not row["solutions"] or split_of(row["problem_id"]) != split:
            continue
        problems.append(
            Problem(
                problem_id=row["problem_id"],
                question=row["question"],
                inputs=row["inputs"],
                outputs=row["outputs"],
                reference=row["solutions"][0]["code"],
            )
        )
    return problems


def sample_problems(split: str, n: int, seed: int = 0, difficulty: str = "interview") -> list[Problem]:
    problems = load_problems(split, difficulty)
    random.Random(seed).shuffle(problems)
    return problems[:n]
