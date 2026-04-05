from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .compat import Action, Observation, State


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    difficulty: TaskDifficulty
    prompt: str = Field(min_length=1)
    buggy_code: str = Field(min_length=1)
    target_symbol: str = Field(min_length=1)
    grader_name: str = Field(min_length=1)
    max_steps: int = Field(default=5, ge=1)
    entrypoint: str = Field(default="solution.py", min_length=1)
    tags: list[str] = Field(default_factory=list)
    public_examples: list[str] = Field(default_factory=list)

    @field_validator("buggy_code")
    @classmethod
    def ensure_trailing_newline(cls, value: str) -> str:
        return value if value.endswith("\n") else f"{value}\n"


class CheckResult(BaseModel):
    name: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)
    passed: bool
    feedback: str = Field(min_length=1)
    category: Literal["syntax", "contract", "tests", "runtime", "timeout"] = (
        "tests"
    )


class GradeReport(BaseModel):
    task_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    compile_success: bool
    tests_passed: int = Field(default=0, ge=0)
    total_tests: int = Field(default=0, ge=0)
    summary: str = Field(min_length=1)
    checks: list[CheckResult] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    error_type: Optional[str] = None
    execution_ms: int = Field(default=0, ge=0)


class RewardSignal(BaseModel):
    reward: float
    current_score: float = Field(ge=0.0, le=1.0)
    previous_score: float = Field(ge=0.0, le=1.0)
    best_score: float = Field(ge=0.0, le=1.0)
    solved: bool = False


class AttemptRecord(BaseModel):
    step_index: int = Field(ge=1)
    score: float = Field(ge=0.0, le=1.0)
    reward: float
    summary: str = Field(min_length=1)
    failing_checks: list[str] = Field(default_factory=list)


class CodeFixAction(Action):
    candidate_code: str = Field(
        min_length=1,
        description="Full corrected Python module for the active task.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional reasoning or patch summary from the agent.",
    )

    @field_validator("candidate_code")
    @classmethod
    def normalize_candidate_code(cls, value: str) -> str:
        normalized = value.strip("\n")
        return f"{normalized}\n"


class CodeReviewObservation(Observation):
    task_id: str = Field(min_length=1)
    difficulty: TaskDifficulty
    title: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    buggy_code: str = Field(min_length=1)
    current_code: str = Field(min_length=1)
    feedback: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    best_score: float = Field(default=0.0, ge=0.0, le=1.0)
    tests_passed: int = Field(default=0, ge=0)
    total_tests: int = Field(default=0, ge=0)
    remaining_steps: int = Field(default=0, ge=0)
    public_examples: list[str] = Field(default_factory=list)
    score_breakdown: list[CheckResult] = Field(default_factory=list)
    reward_signal: Optional[RewardSignal] = None


class CodeReviewState(State):
    task_id: Optional[str] = None
    difficulty: Optional[TaskDifficulty] = None
    title: str = ""
    prompt: str = ""
    buggy_code: str = ""
    current_code: str = ""
    max_steps: int = Field(default=0, ge=0)
    previous_score: float = Field(default=0.0, ge=0.0, le=1.0)
    best_score: float = Field(default=0.0, ge=0.0, le=1.0)
    last_reward: float = 0.0
    last_reward_signal: Optional[RewardSignal] = None
    latest_grade: Optional[GradeReport] = None
    history: list[AttemptRecord] = Field(default_factory=list)
