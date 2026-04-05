from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server.interfaces import Environment  # type: ignore
    try:
        from openenv.core.env_server.http_server import create_app  # type: ignore
    except ImportError:
        from openenv.core.env_server import create_app  # type: ignore
    from openenv.core.env_server.types import (  # type: ignore
        Action,
        EnvironmentMetadata,
        Observation,
        State,
    )

    OPENENV_AVAILABLE = True
except ImportError:
    OPENENV_AVAILABLE = False

    ActT = TypeVar("ActT", bound="Action")
    ObsT = TypeVar("ObsT", bound="Observation")
    StateT = TypeVar("StateT", bound="State")

    class Action(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )
        done: bool = Field(default=False)
        reward: bool | int | float | None = Field(default=None)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        model_config = ConfigDict(
            extra="allow",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )
        episode_id: Optional[str] = Field(default=None)
        step_count: int = Field(default=0, ge=0)

    class EnvironmentMetadata(BaseModel):
        model_config = ConfigDict(extra="forbid")
        name: str
        description: str
        version: Optional[str] = None
        author: Optional[str] = None
        documentation_url: Optional[str] = None
        readme_content: Optional[str] = None

    class Environment(ABC, Generic[ActT, ObsT, StateT]):
        SUPPORTS_CONCURRENT_SESSIONS: bool = False

        def __init__(self, transform: Any = None, rubric: Any = None):
            self.transform = transform
            self.rubric = rubric

        @abstractmethod
        def reset(
            self,
            seed: Optional[int] = None,
            episode_id: Optional[str] = None,
            **kwargs: Any,
        ) -> ObsT:
            raise NotImplementedError

        @abstractmethod
        def step(
            self,
            action: ActT,
            timeout_s: Optional[float] = None,
            **kwargs: Any,
        ) -> ObsT:
            raise NotImplementedError

        @property
        @abstractmethod
        def state(self) -> StateT:
            raise NotImplementedError

        def get_metadata(self) -> EnvironmentMetadata:
            return EnvironmentMetadata(
                name=self.__class__.__name__,
                description=f"{self.__class__.__name__} environment",
                version="1.0.0",
            )

        def close(self) -> None:
            return None

    def create_app(
        env: Callable[[], Environment[Any, Any, Any]],
        action_cls: Type[Action],
        observation_cls: Type[Observation],
        env_name: Optional[str] = None,
        max_concurrent_envs: Optional[int] = None,
        concurrency_config: Optional[Any] = None,
        gradio_builder: Optional[Callable[..., Any]] = None,
    ):
        try:
            from fastapi import FastAPI
        except ImportError as exc:
            raise ImportError(
                "FastAPI is required to serve the fallback HTTP app."
            ) from exc

        app = FastAPI(
            title="OpenEnv Compatibility App",
            version="1.0.0",
            description=(
                "Fallback HTTP app that mirrors the core OpenEnv reset, step, "
                "and state endpoints."
            ),
        )
        environment = env()

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "healthy"}

        @app.get("/metadata")
        def metadata() -> dict[str, Any]:
            return environment.get_metadata().model_dump()

        @app.get("/state")
        def state() -> dict[str, Any]:
            return environment.state.model_dump()

        @app.get("/schema")
        def schema() -> dict[str, Any]:
            return {
                "action": action_cls.model_json_schema(),
                "observation": observation_cls.model_json_schema(),
                "state": environment.state.__class__.model_json_schema(),
            }

        @app.post("/reset")
        def reset(payload: dict[str, Any] | None = None) -> dict[str, Any]:
            payload = payload or {}
            observation = environment.reset(**payload)
            return {
                "observation": observation.model_dump(),
                "reward": observation.reward,
                "done": observation.done,
            }

        @app.post("/step")
        def step(payload: dict[str, Any]) -> dict[str, Any]:
            action = action_cls(**payload.get("action", {}))
            timeout_s = payload.get("timeout_s")
            observation = environment.step(action, timeout_s=timeout_s)
            return {
                "observation": observation.model_dump(),
                "reward": observation.reward,
                "done": observation.done,
            }

        return app
