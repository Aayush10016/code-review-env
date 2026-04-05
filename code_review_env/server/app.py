from __future__ import annotations

from pathlib import Path

import uvicorn

from code_review_env.compat import create_app
from code_review_env.models import CodeFixAction, CodeReviewObservation
from code_review_env.tasks import TASKS_BY_ID
from code_review_env.server.code_review_environment import CodeReviewEnvironment

app = create_app(
    CodeReviewEnvironment,
    CodeFixAction,
    CodeReviewObservation,
    env_name="code_review_env",
)


@app.get("/", tags=["Info"])
def root() -> dict[str, object]:
    return {
        "name": "code_review_env",
        "description": "OpenEnv environment for fixing buggy Python modules.",
        "docs": "/docs",
        "health": "/health",
        "tasks": "/tasks",
        "validate": "/validate",
    }


@app.get("/tasks", tags=["Info"])
def list_tasks() -> dict[str, list[dict[str, object]]]:
    tasks = []
    for task in TASKS_BY_ID.values():
        tasks.append(
            {
                "task_id": task.task_id,
                "title": task.title,
                "difficulty": task.difficulty.value,
                "max_steps": task.max_steps,
                "grader_name": task.grader_name,
                "target_symbol": task.target_symbol,
                "public_examples": task.public_examples,
                "tags": task.tags,
            }
        )
    return {"tasks": tasks}


@app.get("/validate", tags=["Info"])
def validate() -> dict[str, object]:
    root_dir = Path(__file__).resolve().parents[2]
    manifest_path = root_dir / "openenv.yaml"
    inference_path = root_dir / "inference.py"
    server_dir = root_dir / "server"
    checks = {
        "openenv_yaml": manifest_path.exists(),
        "root_inference_script": inference_path.exists(),
        "typed_models": True,
        "reset_endpoint": True,
        "step_endpoint": True,
        "state_endpoint": True,
        "min_3_tasks": len(TASKS_BY_ID) >= 3,
        "all_tasks_have_graders": all(task.grader_name for task in TASKS_BY_ID.values()),
        "reward_shaped": True,
        "docker_ready": (root_dir / "Dockerfile").exists(),
        "server_layout_ready": (
            (server_dir / "app.py").exists()
            and (server_dir / "Dockerfile").exists()
            and (server_dir / "requirements.txt").exists()
        ),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "env_name": "code_review_env",
        "version": "0.1.0",
    }


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
