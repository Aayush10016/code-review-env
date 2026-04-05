from __future__ import annotations

import random
from textwrap import dedent

from .models import TaskDescriptor, TaskDifficulty


TASKS_BY_ID: dict[str, TaskDescriptor] = {
    "easy_dedupe": TaskDescriptor(
        task_id="easy_dedupe",
        title="Order-Preserving Deduplication",
        difficulty=TaskDifficulty.EASY,
        prompt=(
            "Fix the helper so it removes duplicates while preserving the first "
            "occurrence order. Submit the full corrected Python module."
        ),
        buggy_code=dedent(
            """
            from typing import Iterable, List, TypeVar

            T = TypeVar("T")


            def dedupe_preserve_order(items: Iterable[T]) -> List[T]:
                \"\"\"Return the first occurrence of each item while preserving input order.\"\"\"
                return list(set(items))
            """
        ),
        target_symbol="dedupe_preserve_order",
        grader_name="grade_easy_dedupe",
        max_steps=4,
        tags=["collections", "data-cleaning", "utility"],
        public_examples=[
            "dedupe_preserve_order(['a', 'b', 'a', 'c']) -> ['a', 'b', 'c']",
            "dedupe_preserve_order((1, 1, 2, 3, 2)) -> [1, 2, 3]",
        ],
    ),
    "medium_merge_intervals": TaskDescriptor(
        task_id="medium_merge_intervals",
        title="Merge Scheduling Intervals",
        difficulty=TaskDifficulty.MEDIUM,
        prompt=(
            "Fix the interval merge utility used by a scheduling service. The "
            "function must sort intervals correctly, merge overlaps and touching "
            "ranges, and return tuples. Submit the full corrected module."
        ),
        buggy_code=dedent(
            """
            from typing import Iterable, List, Tuple

            Interval = Tuple[int, int]


            def merge_intervals(intervals: Iterable[Interval]) -> List[Interval]:
                intervals = list(intervals)
                if not intervals:
                    return []

                intervals = sorted(intervals, key=lambda pair: pair[1])
                merged = [list(intervals[0])]

                for start, end in intervals[1:]:
                    last_start, last_end = merged[-1]
                    if start < last_end:
                        merged[-1][1] = end
                    else:
                        merged.append([start, end])

                return merged
            """
        ),
        target_symbol="merge_intervals",
        grader_name="grade_medium_merge_intervals",
        max_steps=5,
        tags=["scheduling", "intervals", "backend"],
        public_examples=[
            "merge_intervals([(1, 3), (2, 6), (8, 10)]) -> [(1, 6), (8, 10)]",
            "merge_intervals([(1, 3), (3, 5)]) -> [(1, 5)]",
        ],
    ),
    "hard_lru_cache": TaskDescriptor(
        task_id="hard_lru_cache",
        title="Repair LRU Cache Eviction",
        difficulty=TaskDifficulty.HARD,
        prompt=(
            "Fix the LRU cache implementation so `get` refreshes recency, `put` "
            "updates existing keys correctly, and eviction removes the least "
            "recently used entry. Submit the full corrected module."
        ),
        buggy_code=dedent(
            """
            from collections import OrderedDict


            class LRUCache:
                def __init__(self, capacity: int):
                    self.capacity = capacity
                    self.data = OrderedDict()

                def get(self, key):
                    return self.data.get(key, -1)

                def put(self, key, value):
                    if key in self.data:
                        self.data[key] = value
                        return

                    self.data[key] = value
                    if len(self.data) > self.capacity:
                        self.data.popitem()
            """
        ),
        target_symbol="LRUCache",
        grader_name="grade_hard_lru_cache",
        max_steps=6,
        tags=["cache", "data-structures", "systems"],
        public_examples=[
            "After get(1), key 1 should become the most recently used entry.",
            "When capacity is exceeded, evict the least recently used key.",
        ],
    ),
}

TASKS_BY_DIFFICULTY: dict[TaskDifficulty, list[TaskDescriptor]] = {
    difficulty: [
        task for task in TASKS_BY_ID.values() if task.difficulty == difficulty
    ]
    for difficulty in TaskDifficulty
}


def get_task(task_id: str) -> TaskDescriptor:
    try:
        return TASKS_BY_ID[task_id]
    except KeyError as exc:
        available = ", ".join(sorted(TASKS_BY_ID))
        raise ValueError(
            f"Unknown task_id '{task_id}'. Available tasks: {available}."
        ) from exc


def select_task(
    rng: random.Random,
    *,
    task_id: str | None = None,
    difficulty: TaskDifficulty | str | None = None,
) -> TaskDescriptor:
    if task_id is not None:
        return get_task(task_id)

    if difficulty is not None and not isinstance(difficulty, TaskDifficulty):
        difficulty = TaskDifficulty(difficulty)

    if difficulty is None:
        return rng.choice(list(TASKS_BY_ID.values()))

    candidates = TASKS_BY_DIFFICULTY[difficulty]
    return rng.choice(candidates)
