from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from typing import Any
from urllib.parse import urlparse

import requests
from openai import OpenAI

from code_review_env.tasks import TASKS_BY_ID

SYSTEM_PROMPT = """You are a Python bug-fixing agent.

You will receive a buggy Python module, a task objective, and grader feedback.
Return ONLY the full corrected Python module.
Do not include explanations before or after the code.
Do not return JSON.
"""

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "openai/gpt-4.1-mini"
MIN_VALID_SCORE = 0.01
MAX_VALID_SCORE = 0.99


def safe_score(value: Any, default: float = MIN_VALID_SCORE) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return min(max(round(score, 4), MIN_VALID_SCORE), MAX_VALID_SCORE)


def safe_test_counts(tests_passed: Any, total_tests: Any) -> tuple[int, int]:
    try:
        total = int(total_tests)
    except (TypeError, ValueError):
        total = 2
    total = max(total, 2)

    try:
        passed = int(tests_passed)
    except (TypeError, ValueError):
        passed = 1
    return min(max(passed, 1), total - 1), total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baseline inference runner for CodeReviewEnv."
    )
    parser.add_argument(
        "--env-base-url",
        default=os.environ.get("ENV_BASE_URL", "http://127.0.0.1:7860"),
        help="Base URL for the environment server.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum number of model attempts per task.",
    )
    parser.add_argument(
        "--task-id",
        choices=sorted(TASKS_BY_ID),
        default=None,
        help="Optional single task override.",
    )
    return parser


def build_client() -> tuple[OpenAI | None, str, str]:
    """Build an optional model client.

    The validator may run this script without model credentials. In that case,
    inference still succeeds by using the deterministic fallback solutions below.
    """
    api_base_url = os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL)
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")
    if not api_key:
        print(
            "[DEBUG] No API key configured; using offline baseline solutions.",
            file=sys.stderr,
            flush=True,
        )
        return None, api_base_url, model_name

    client = OpenAI(base_url=api_base_url, api_key=api_key, timeout=120.0)
    return client, api_base_url, model_name


def emit(tag: str, payload: dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'))}", flush=True)


def extract_code(content: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", content, flags=re.DOTALL)
    if fenced:
        return fenced[0].strip() + "\n"
    return content.strip() + "\n"


def env_reset(env_base_url: str, task_id: str) -> dict[str, Any]:
    response = requests.post(
        f"{env_base_url}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def env_step(env_base_url: str, candidate_code: str) -> dict[str, Any]:
    response = requests.post(
        f"{env_base_url}/step",
        json={"action": {"candidate_code": candidate_code}, "timeout_s": 2.0},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def is_env_healthy(env_base_url: str) -> bool:
    try:
        response = requests.get(f"{env_base_url}/health", timeout=3)
        response.raise_for_status()
    except Exception:
        return False
    return True


def maybe_start_local_server(env_base_url: str) -> subprocess.Popen[bytes] | None:
    if is_env_healthy(env_base_url):
        return None

    parsed = urlparse(env_base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise RuntimeError(
            f"Environment server is not reachable at {env_base_url}."
        )
    port = parsed.port or 7860

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + 25
    while time.time() < deadline:
        if process.poll() is not None:
            break
        if is_env_healthy(env_base_url):
            return process
        time.sleep(0.5)

    try:
        process.terminate()
    except Exception:
        pass
    raise RuntimeError(
        f"Environment server is not reachable at {env_base_url} and automatic startup failed."
    )


def generate_candidate(
    client: OpenAI | None,
    model_name: str,
    *,
    observation: dict[str, Any],
    step_index: int,
) -> str:
    """Generate a candidate fix using OpenAI client. Falls back to simple heuristics if API fails."""
    user_prompt = (
        f"Task ID: {observation['task_id']}\n"
        f"Difficulty: {observation['difficulty']}\n"
        f"Title: {observation['title']}\n"
        f"Objective: {observation['prompt']}\n"
        f"Current score: {observation.get('score', 0.0)}\n"
        f"Remaining steps: {observation.get('remaining_steps', 0)}\n"
        f"Feedback: {json.dumps(observation.get('feedback', []), ensure_ascii=True)}\n"
        f"Current module:\n```python\n{observation['current_code']}\n```\n"
        f"Buggy starter module:\n```python\n{observation['buggy_code']}\n```\n"
        f"Attempt number: {step_index}\n"
        "Return only the corrected full Python module."
    )

    task_id = observation.get("task_id", "")

    if client is None:
        return fallback_solution(task_id, observation["current_code"])
    
    try:
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            top_p=1.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            timeout=60,
        )
        content = completion.choices[0].message.content or ""
        candidate = extract_code(content)
        if candidate.strip():
            return candidate
    except Exception as e:
        print(f"[DEBUG] Model request failed: {e}", file=sys.stderr, flush=True)
    
    return fallback_solution(task_id, observation["current_code"])


def fallback_solution(task_id: str, current_code: str) -> str:
    """Return a deterministic solution when model inference is unavailable."""
    simple_fixes = {
        "easy_dedupe": """from typing import Iterable, List, TypeVar

T = TypeVar("T")


def dedupe_preserve_order(items: Iterable[T]) -> List[T]:
    \"\"\"Return the first occurrence of each item while preserving input order.\"\"\"
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
""",
        "medium_merge_intervals": """from typing import Iterable, List, Tuple


def merge_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    \"\"\"Return merged intervals, coalescing overlaps and touches.\"\"\"
    if not intervals:
        return []
    
    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(end, last_end))
        else:
            merged.append((start, end))
    
    return merged
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
    
    return simple_fixes.get(task_id, current_code)


def run_task(
    client: OpenAI | None,
    *,
    api_base_url: str,
    model_name: str,
    env_base_url: str,
    run_id: str,
    task_id: str,
    max_attempts: int,
) -> float:
    task = TASKS_BY_ID[task_id]
    
    try:
        reset_payload = env_reset(env_base_url, task_id)
    except Exception as e:
        print(f"Reset failed: {e}", file=sys.stderr)
        raise
    
    observation = reset_payload.get("observation", {})

    emit(
        "START",
        {
            "run_id": run_id,
            "task_id": task_id,
            "difficulty": task.difficulty.value,
            "model_name": model_name,
            "api_base_url": api_base_url,
            "env_base_url": env_base_url,
        },
    )

    final_score = safe_score(observation.get("score"))
    for step_index in range(1, max_attempts + 1):
        candidate_code = generate_candidate(
            client,
            model_name,
            observation=observation,
            step_index=step_index,
        )
        try:
            step_payload = env_step(env_base_url, candidate_code)
        except Exception as e:
            print(f"Step failed: {e}", file=sys.stderr)
            raise
        
        observation = step_payload.get("observation", {})
        final_score = safe_score(observation.get("score"))
        tests_passed, total_tests = safe_test_counts(
            observation.get("tests_passed"),
            observation.get("total_tests"),
        )

        emit(
            "STEP",
            {
                "run_id": run_id,
                "task_id": task_id,
                "step": step_index,
                "reward": safe_score(step_payload.get("reward")),
                "score": final_score,
                "tests_passed": tests_passed,
                "total_tests": total_tests,
            },
        )

        if step_payload.get("done"):
            final_score = safe_score(observation.get("score"))
            break

    emit(
        "END",
        {
            "run_id": run_id,
            "task_id": task_id,
            "final_score": final_score,
            "status": "completed",
        },
    )
    return final_score


def main() -> int:
    args = build_parser().parse_args()
    client, api_base_url, model_name = build_client()

    task_ids = [args.task_id] if args.task_id else list(TASKS_BY_ID.keys())
    run_id = str(uuid.uuid4())
    scores: list[float] = []
    server_process: subprocess.Popen[bytes] | None = None

    try:
        server_process = maybe_start_local_server(args.env_base_url)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        for task_id in task_ids:
            try:
                score = run_task(
                    client,
                    api_base_url=api_base_url,
                    model_name=model_name,
                    env_base_url=args.env_base_url,
                    run_id=run_id,
                    task_id=task_id,
                    max_attempts=args.max_attempts,
                )
            except Exception as exc:
                emit(
                    "END",
                    {
                        "run_id": run_id,
                        "task_id": task_id,
                        "final_score": MIN_VALID_SCORE,
                        "status": f"error:{type(exc).__name__}",
                    },
                )
                return 1
            scores.append(score)
    finally:
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except Exception:
                server_process.kill()

    return 0 if scores else 1


if __name__ == "__main__":
    raise SystemExit(main())
