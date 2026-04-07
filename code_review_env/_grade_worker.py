from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable


def record(
    checks: list[dict[str, Any]],
    *,
    name: str,
    weight: float,
    passed: bool,
    feedback: str,
    category: str,
) -> None:
    checks.append(
        {
            "name": name,
            "weight": weight,
            "passed": passed,
            "feedback": feedback,
            "category": category,
        }
    )


def safe_test(
    checks: list[dict[str, Any]],
    *,
    name: str,
    weight: float,
    success_message: str,
    failure_message: str,
    fn: Callable[[], bool],
    category: str = "tests",
) -> bool:
    try:
        passed = bool(fn())
        detail = ""
    except Exception as exc:
        passed = False
        detail = f" ({type(exc).__name__}: {exc})"

    record(
        checks,
        name=name,
        weight=weight,
        passed=passed,
        feedback=success_message if passed else f"{failure_message}{detail}",
        category=category,
    )
    return passed


def fail_pending(
    checks: list[dict[str, Any]],
    suite: list[tuple[str, float, str]],
    reason: str,
) -> None:
    for name, weight, category in suite:
        record(
            checks,
            name=name,
            weight=weight,
            passed=False,
            feedback=reason,
            category=category,
        )


def grade_easy(
    namespace: dict[str, Any],
    checks: list[dict[str, Any]],
) -> tuple[int, int]:
    pending = [
        ("function exists", 0.10, "contract"),
        ("preserves first-occurrence order", 0.35, "tests"),
        ("handles empty iterables", 0.15, "tests"),
        ("accepts general iterables", 0.15, "tests"),
        ("returns a list result", 0.15, "tests"),
    ]
    fn = namespace.get("dedupe_preserve_order")
    if not callable(fn):
        fail_pending(
            checks,
            pending,
            "Expected a callable named dedupe_preserve_order(items).",
        )
        return 0, 4

    record(
        checks,
        name="function exists",
        weight=0.10,
        passed=True,
        feedback="Found callable dedupe_preserve_order.",
        category="contract",
    )

    tests_passed = 0
    if safe_test(
        checks,
        name="preserves first-occurrence order",
        weight=0.35,
        success_message="Duplicate items are removed without changing first-seen order.",
        failure_message="The helper still changes element order or keeps the wrong occurrences.",
        fn=lambda: fn(["a", "b", "a", "c", "b"]) == ["a", "b", "c"],
    ):
        tests_passed += 1
    if safe_test(
        checks,
        name="handles empty iterables",
        weight=0.15,
        success_message="Empty inputs return an empty list.",
        failure_message="Empty input handling is still incorrect.",
        fn=lambda: fn([]) == [],
    ):
        tests_passed += 1
    if safe_test(
        checks,
        name="accepts general iterables",
        weight=0.15,
        success_message="The helper works for non-list iterables as well.",
        failure_message="The helper fails for tuple input or loses expected values.",
        fn=lambda: fn((1, 1, 2, 3, 2)) == [1, 2, 3],
    ):
        tests_passed += 1
    if safe_test(
        checks,
        name="returns a list result",
        weight=0.15,
        success_message="The return type is a list of unique first occurrences.",
        failure_message="The helper should return a list with first occurrences only.",
        fn=lambda: isinstance(fn(["x", "x", "y"]), list)
        and fn(["x", "x", "y"]) == ["x", "y"],
    ):
        tests_passed += 1

    return tests_passed, 4


def grade_medium(
    namespace: dict[str, Any],
    checks: list[dict[str, Any]],
) -> tuple[int, int]:
    pending = [
        ("function exists", 0.10, "contract"),
        ("merges overlapping intervals", 0.25, "tests"),
        ("merges touching intervals", 0.20, "tests"),
        ("handles nested intervals", 0.15, "tests"),
        ("returns sorted output", 0.10, "tests"),
        ("returns tuples", 0.10, "tests"),
    ]
    fn = namespace.get("merge_intervals")
    if not callable(fn):
        fail_pending(
            checks,
            pending,
            "Expected a callable named merge_intervals(intervals).",
        )
        return 0, 5

    record(
        checks,
        name="function exists",
        weight=0.10,
        passed=True,
        feedback="Found callable merge_intervals.",
        category="contract",
    )

    tests_passed = 0
    if safe_test(
        checks,
        name="merges overlapping intervals",
        weight=0.25,
        success_message="Overlapping intervals are merged correctly.",
        failure_message="Overlapping ranges are still merged incorrectly.",
        fn=lambda: fn([(1, 3), (2, 6), (8, 10), (15, 18)])
        == [(1, 6), (8, 10), (15, 18)],
    ):
        tests_passed += 1
    if safe_test(
        checks,
        name="merges touching intervals",
        weight=0.20,
        success_message="Touching ranges are treated as a single merged interval.",
        failure_message="Touching ranges should merge into one interval.",
        fn=lambda: fn([(5, 7), (1, 3), (3, 5)]) == [(1, 7)],
    ):
        tests_passed += 1
    if safe_test(
        checks,
        name="handles nested intervals",
        weight=0.15,
        success_message="Nested intervals collapse to the containing range.",
        failure_message="Nested ranges are not handled correctly.",
        fn=lambda: fn([(1, 10), (2, 3), (4, 8)]) == [(1, 10)],
    ):
        tests_passed += 1
    if safe_test(
        checks,
        name="returns sorted output",
        weight=0.10,
        success_message="Output intervals are sorted by start time.",
        failure_message="Output should be sorted by start time.",
        fn=lambda: fn([(8, 9), (1, 2), (4, 5)]) == [(1, 2), (4, 5), (8, 9)],
    ):
        tests_passed += 1
    if safe_test(
        checks,
        name="returns tuples",
        weight=0.10,
        success_message="Merged intervals are returned as tuples.",
        failure_message="Each merged interval should be returned as a tuple.",
        fn=lambda: all(
            isinstance(interval, tuple)
            for interval in fn([(1, 2), (2, 4), (7, 9)])
        ),
    ):
        tests_passed += 1

    return tests_passed, 5


def grade_hard(
    namespace: dict[str, Any],
    checks: list[dict[str, Any]],
) -> tuple[int, int]:
    pending = [
        ("class exists", 0.10, "contract"),
        ("supports basic put/get", 0.15, "tests"),
        ("refreshes recency on get", 0.25, "tests"),
        ("updates existing keys correctly", 0.20, "tests"),
        ("handles capacity one", 0.10, "tests"),
        ("evicts least recently used key", 0.10, "tests"),
    ]
    cls = namespace.get("LRUCache")
    if not isinstance(cls, type):
        fail_pending(
            checks,
            pending,
            "Expected a class named LRUCache.",
        )
        return 0, 5

    record(
        checks,
        name="class exists",
        weight=0.10,
        passed=True,
        feedback="Found class LRUCache.",
        category="contract",
    )

    def basic_case() -> bool:
        cache = cls(2)
        cache.put(1, "a")
        cache.put(2, "b")
        return cache.get(1) == "a" and cache.get(2) == "b"

    def refresh_case() -> bool:
        cache = cls(2)
        cache.put(1, "a")
        cache.put(2, "b")
        cache.get(1)
        cache.put(3, "c")
        return cache.get(1) == "a" and cache.get(2) == -1 and cache.get(3) == "c"

    def update_case() -> bool:
        cache = cls(2)
        cache.put(1, "a")
        cache.put(2, "b")
        cache.put(1, "updated")
        cache.put(3, "c")
        return (
            cache.get(1) == "updated"
            and cache.get(2) == -1
            and cache.get(3) == "c"
        )

    def capacity_one_case() -> bool:
        cache = cls(1)
        cache.put("x", 10)
        cache.put("y", 20)
        return cache.get("x") == -1 and cache.get("y") == 20

    def eviction_case() -> bool:
        cache = cls(2)
        cache.put("alpha", 1)
        cache.put("beta", 2)
        cache.put("gamma", 3)
        return cache.get("alpha") == -1 and cache.get("beta") == 2

    tests_passed = 0
    if safe_test(
        checks,
        name="supports basic put/get",
        weight=0.15,
        success_message="Basic put/get behavior works.",
        failure_message="Basic put/get behavior is still incorrect.",
        fn=basic_case,
    ):
        tests_passed += 1
    if safe_test(
        checks,
        name="refreshes recency on get",
        weight=0.25,
        success_message="Calling get refreshes the accessed key's recency.",
        failure_message="Recent access is not updating eviction order.",
        fn=refresh_case,
    ):
        tests_passed += 1
    if safe_test(
        checks,
        name="updates existing keys correctly",
        weight=0.20,
        success_message="Updating an existing key preserves correct cache behavior.",
        failure_message="Updating an existing key still breaks recency or capacity handling.",
        fn=update_case,
    ):
        tests_passed += 1
    if safe_test(
        checks,
        name="handles capacity one",
        weight=0.10,
        success_message="Capacity-one caches evict the old entry correctly.",
        failure_message="Capacity-one eviction behavior is still wrong.",
        fn=capacity_one_case,
    ):
        tests_passed += 1
    if safe_test(
        checks,
        name="evicts least recently used key",
        weight=0.10,
        success_message="Eviction removes the least recently used entry.",
        failure_message="Eviction is still removing the wrong key.",
        fn=eviction_case,
    ):
        tests_passed += 1

    return tests_passed, 5


GRADERS: dict[str, Callable[[dict[str, Any], list[dict[str, Any]]], tuple[int, int]]] = {
    "easy_dedupe": grade_easy,
    "medium_merge_intervals": grade_medium,
    "hard_lru_cache": grade_hard,
}


def main() -> None:
    task_id = sys.argv[1]
    solution_path = Path(sys.argv[2])
    grader = GRADERS[task_id]

    candidate_code = solution_path.read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": "solution"}
    checks: list[dict[str, Any]] = []
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    compile_success = False
    tests_passed = 0
    total_tests = 0
    error_type: str | None = None
    start = time.perf_counter()

    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
        stderr_buffer
    ):
        try:
            exec(compile(candidate_code, str(solution_path), "exec"), namespace)
        except Exception as exc:
            error_type = type(exc).__name__
            record(
                checks,
                name="code compiles",
                weight=0.10,
                passed=False,
                feedback=f"Candidate failed to compile or import: {type(exc).__name__}: {exc}",
                category="syntax",
            )
            if task_id == "easy_dedupe":
                fail_pending(
                    checks,
                    [
                        ("function exists", 0.10, "contract"),
                        ("preserves first-occurrence order", 0.35, "tests"),
                        ("handles empty iterables", 0.15, "tests"),
                        ("accepts general iterables", 0.15, "tests"),
                        ("returns a list result", 0.15, "tests"),
                    ],
                    "Compilation failed, so functional checks could not run.",
                )
                total_tests = 4
            elif task_id == "medium_merge_intervals":
                fail_pending(
                    checks,
                    [
                        ("function exists", 0.10, "contract"),
                        ("merges overlapping intervals", 0.25, "tests"),
                        ("merges touching intervals", 0.20, "tests"),
                        ("handles nested intervals", 0.15, "tests"),
                        ("returns sorted output", 0.10, "tests"),
                        ("returns tuples", 0.10, "tests"),
                    ],
                    "Compilation failed, so functional checks could not run.",
                )
                total_tests = 5
            else:
                fail_pending(
                    checks,
                    [
                        ("class exists", 0.10, "contract"),
                        ("supports basic put/get", 0.15, "tests"),
                        ("refreshes recency on get", 0.25, "tests"),
                        ("updates existing keys correctly", 0.20, "tests"),
                        ("handles capacity one", 0.10, "tests"),
                        ("evicts least recently used key", 0.10, "tests"),
                    ],
                    "Compilation failed, so functional checks could not run.",
                )
                total_tests = 5
        else:
            compile_success = True
            record(
                checks,
                name="code compiles",
                weight=0.10,
                passed=True,
                feedback="Candidate module compiled successfully.",
                category="syntax",
            )
            tests_passed, total_tests = grader(namespace, checks)

    total_weight = sum(check["weight"] for check in checks)
    score = 0.0
    if total_weight:
        score = round(
            sum(check["weight"] for check in checks if check["passed"]) / total_weight,
            4,
        )
        if score >= 1.0:
            score = 0.99

    failing_feedback = [check["feedback"] for check in checks if not check["passed"]]
    summary = (
        "All rubric checks passed."
        if score >= 0.99
        else f"Passed {tests_passed}/{total_tests} hidden functional checks."
    )
    if not compile_success:
        summary = "Candidate failed to compile, so no functional checks could complete."

    result = {
        "task_id": task_id,
        "score": score,
        "compile_success": compile_success,
        "tests_passed": tests_passed,
        "total_tests": total_tests,
        "summary": summary,
        "checks": checks,
        "feedback": list(dict.fromkeys(failing_feedback))[:5],
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
        "error_type": error_type,
        "execution_ms": int((time.perf_counter() - start) * 1000),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
