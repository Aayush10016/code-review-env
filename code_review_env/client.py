from __future__ import annotations

from typing import Any

from .models import CodeFixAction, CodeReviewObservation, CodeReviewState

try:
    from openenv.core.client_types import StepResult  # type: ignore
    from openenv.core.env_client import EnvClient  # type: ignore

    class CodeReviewEnv(
        EnvClient[CodeFixAction, CodeReviewObservation, CodeReviewState]
    ):
        def _step_payload(self, action: CodeFixAction) -> dict[str, Any]:
            return action.model_dump(exclude_none=True)

        def _parse_result(self, payload: dict[str, Any]) -> StepResult[CodeReviewObservation]:
            observation = CodeReviewObservation(**payload["observation"])
            return StepResult(
                observation=observation,
                reward=payload.get("reward"),
                done=payload.get("done", observation.done),
            )

        def _parse_state(self, payload: dict[str, Any]) -> CodeReviewState:
            return CodeReviewState(**payload)

except ImportError:
    import requests
    from pydantic import BaseModel

    class StepResult(BaseModel):
        observation: CodeReviewObservation
        reward: float | int | bool | None = None
        done: bool = False

    class CodeReviewEnv:
        def __init__(self, base_url: str):
            self.base_url = base_url.rstrip("/")

        def reset(self, **kwargs: Any) -> StepResult:
            response = requests.post(
                f"{self.base_url}/reset",
                json=kwargs,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            return StepResult(
                observation=CodeReviewObservation(**payload["observation"]),
                reward=payload.get("reward"),
                done=payload.get("done", False),
            )

        def step(self, action: CodeFixAction, **kwargs: Any) -> StepResult:
            payload = {"action": action.model_dump(exclude_none=True), **kwargs}
            response = requests.post(
                f"{self.base_url}/step",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return StepResult(
                observation=CodeReviewObservation(**data["observation"]),
                reward=data.get("reward"),
                done=data.get("done", False),
            )

        def state(self) -> CodeReviewState:
            response = requests.get(f"{self.base_url}/state", timeout=30)
            response.raise_for_status()
            return CodeReviewState(**response.json())

        def close(self) -> None:
            return None

