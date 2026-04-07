from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from .models import CheckResult, GradeReport, TaskDescriptor
from .tasks import get_task

WORKER_PATH = Path(__file__).with_name("_grade_worker.py")
MIN_VALID_SCORE = 0.01
MAX_VALID_SCORE = 0.99


def _clamp_score(score: float) -> float:
    return min(max(round(score, 4), MIN_VALID_SCORE), MAX_VALID_SCORE)


def _timeout_report(task: TaskDescriptor, timeout_s: float) -> GradeReport:
    return GradeReport(
        task_id=task.task_id,
        score=MIN_VALID_SCORE,
        compile_success=False,
        tests_passed=0,
        total_tests=0,
        summary="Candidate execution timed out during grading.",
        checks=[
            CheckResult(
                name="grading timeout",
                weight=1.0,
                passed=False,
                feedback=(
                    f"Grading exceeded the timeout budget of {timeout_s:.2f} seconds."
                ),
                category="timeout",
            )
        ],
        feedback=[
            f"Grading exceeded the timeout budget of {timeout_s:.2f} seconds."
        ],
        error_type="TimeoutExpired",
        execution_ms=int(timeout_s * 1000),
    )


def _worker_failure_report(task: TaskDescriptor, stderr: str) -> GradeReport:
    return GradeReport(
        task_id=task.task_id,
        score=MIN_VALID_SCORE,
        compile_success=False,
        tests_passed=0,
        total_tests=0,
        summary="The grader crashed before producing a valid report.",
        checks=[
            CheckResult(
                name="grader runtime",
                weight=1.0,
                passed=False,
                feedback="The grading worker failed unexpectedly.",
                category="runtime",
            )
        ],
        feedback=["The grading worker failed unexpectedly."],
        stderr=stderr,
        error_type="GraderRuntimeError",
    )


def _grade_with_worker(
    task: TaskDescriptor,
    candidate_code: str,
    timeout_s: float = 2.0,
) -> GradeReport:
    with tempfile.TemporaryDirectory(prefix=f"{task.task_id}_") as temp_dir:
        temp_path = Path(temp_dir)
        solution_path = temp_path / task.entrypoint
        solution_path.write_text(candidate_code, encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, str(WORKER_PATH), task.task_id, str(solution_path)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _timeout_report(task, timeout_s)

        stdout = completed.stdout.strip()
        if not stdout:
            return _worker_failure_report(task, completed.stderr)

        try:
            payload = json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError:
            return _worker_failure_report(task, completed.stderr or stdout)

    payload["score"] = _clamp_score(float(payload.get("score", 0.0)))
    payload["checks"] = [CheckResult(**check) for check in payload["checks"]]
    return GradeReport(**payload)


def grade_easy_dedupe(
    candidate_code: str,
    *,
    task: TaskDescriptor | None = None,
    timeout_s: float = 2.0,
) -> GradeReport:
    return _grade_with_worker(task or get_task("easy_dedupe"), candidate_code, timeout_s)


def grade_medium_merge_intervals(
    candidate_code: str,
    *,
    task: TaskDescriptor | None = None,
    timeout_s: float = 2.0,
) -> GradeReport:
    return _grade_with_worker(
        task or get_task("medium_merge_intervals"),
        candidate_code,
        timeout_s,
    )


def grade_hard_lru_cache(
    candidate_code: str,
    *,
    task: TaskDescriptor | None = None,
    timeout_s: float = 2.0,
) -> GradeReport:
    return _grade_with_worker(
        task or get_task("hard_lru_cache"),
        candidate_code,
        timeout_s,
    )


GRADER_REGISTRY: dict[str, Callable[..., GradeReport]] = {
    "grade_easy_dedupe": grade_easy_dedupe,
    "grade_medium_merge_intervals": grade_medium_merge_intervals,
    "grade_hard_lru_cache": grade_hard_lru_cache,
}


def grade_task(
    task: TaskDescriptor,
    candidate_code: str,
    *,
    timeout_s: float = 2.0,
) -> GradeReport:
    grader = GRADER_REGISTRY[task.grader_name]
    return grader(candidate_code, task=task, timeout_s=timeout_s)
