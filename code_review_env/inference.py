from __future__ import annotations

import argparse
import json
from pathlib import Path

from .environment import CodeReviewEnvironment
from .models import CodeFixAction
from .tasks import TASKS_BY_ID

BASELINE_SOLUTIONS: dict[str, str] = {
    "easy_dedupe": """from typing import Iterable, List, TypeVar

T = TypeVar("T")


def dedupe_preserve_order(items: Iterable[T]) -> List[T]:
    \"\"\"Return the first occurrence of each item while preserving input order.\"\"\"
    seen = set()
    ordered: List[T] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
""",
    "medium_merge_intervals": """from typing import Iterable, List, Tuple

Interval = Tuple[int, int]


def merge_intervals(intervals: Iterable[Interval]) -> List[Interval]:
    ordered = sorted(list(intervals), key=lambda pair: pair[0])
    if not ordered:
        return []

    merged: List[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged]
""",
    "hard_lru_cache": """from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return -1
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value):
        if key in self.data:
            self.data[key] = value
            self.data.move_to_end(key)
        else:
            self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
""",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local or HTTP inference demo against CodeReviewEnv."
    )
    parser.add_argument(
        "--mode",
        choices=["local", "http"],
        default="local",
        help="Interact with the environment directly or through the HTTP server.",
    )
    parser.add_argument(
        "--task-id",
        choices=sorted(TASKS_BY_ID),
        default="easy_dedupe",
        help="Task to run.",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Optional difficulty hint used when task_id is not provided.",
    )
    parser.add_argument(
        "--candidate-file",
        type=Path,
        default=None,
        help="Optional path to a Python module to submit instead of the baseline solution.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for HTTP mode.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=2.0,
        help="Timeout passed to the environment step.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the available tasks and exit.",
    )
    return parser


def load_candidate(task_id: str, candidate_file: Path | None) -> str:
    if candidate_file is None:
        return BASELINE_SOLUTIONS[task_id]
    return candidate_file.read_text(encoding="utf-8")


def print_summary(label: str, payload: dict) -> None:
    print(f"\n[{label}]")
    print(json.dumps(payload, indent=2))


def run_local(args: argparse.Namespace) -> None:
    env = CodeReviewEnvironment(default_timeout_s=args.timeout_s)
    observation = env.reset(task_id=args.task_id, difficulty=args.difficulty)
    print_summary("reset", observation.model_dump())

    candidate = load_candidate(args.task_id, args.candidate_file)
    result = env.step(CodeFixAction(candidate_code=candidate), timeout_s=args.timeout_s)
    print_summary("step", result.model_dump())
    print_summary("state", env.state.model_dump())


def run_http(args: argparse.Namespace) -> None:
    import requests

    reset_response = requests.post(
        f"{args.base_url}/reset",
        json={"task_id": args.task_id, "difficulty": args.difficulty},
        timeout=10,
    )
    reset_response.raise_for_status()
    print_summary("reset", reset_response.json())

    candidate = load_candidate(args.task_id, args.candidate_file)
    step_response = requests.post(
        f"{args.base_url}/step",
        json={
            "action": CodeFixAction(candidate_code=candidate).model_dump(),
            "timeout_s": args.timeout_s,
        },
        timeout=20,
    )
    step_response.raise_for_status()
    print_summary("step", step_response.json())

    state_response = requests.get(f"{args.base_url}/state", timeout=10)
    state_response.raise_for_status()
    print_summary("state", state_response.json())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_tasks:
        print(
            json.dumps(
                {
                    task_id: task.model_dump()
                    for task_id, task in TASKS_BY_ID.items()
                },
                indent=2,
            )
        )
        return

    if args.mode == "local":
        run_local(args)
        return

    run_http(args)


if __name__ == "__main__":
    main()
