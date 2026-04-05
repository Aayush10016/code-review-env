from __future__ import annotations

import random
import uuid
from typing import Optional

from .compat import Environment, EnvironmentMetadata
from .graders import grade_task
from .models import (
    AttemptRecord,
    CodeFixAction,
    CodeReviewObservation,
    CodeReviewState,
    RewardSignal,
    TaskDescriptor,
    TaskDifficulty,
)
from .tasks import select_task


class CodeReviewEnvironment(
    Environment[CodeFixAction, CodeReviewObservation, CodeReviewState]
):
    SUPPORTS_CONCURRENT_SESSIONS = False
    
    # Class-level state to persist across instance creations
    _class_task: TaskDescriptor | None = None
    _class_state: CodeReviewState | None = None

    def __init__(self, default_timeout_s: float = 2.0):
        super().__init__()
        self.default_timeout_s = default_timeout_s
        self._rng = random.Random()
        # Use class-level state if available
        self._task = self.__class__._class_task
        self._state = self.__class__._class_state or CodeReviewState()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str | None = None,
        difficulty: TaskDifficulty | str | None = None,
        max_steps: int | None = None,
        **kwargs,
    ) -> CodeReviewObservation:
        if seed is not None:
            self._rng.seed(seed)

        self._task = select_task(
            self._rng,
            task_id=task_id,
            difficulty=difficulty,
        )
        
        # Save to class-level for persistence across instances
        self.__class__._class_task = self._task

        self._state = CodeReviewState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            title=self._task.title,
            prompt=self._task.prompt,
            buggy_code=self._task.buggy_code,
            current_code=self._task.buggy_code,
            max_steps=max_steps or self._task.max_steps,
            previous_score=0.0,
            best_score=0.0,
            last_reward=0.0,
            last_reward_signal=None,
            latest_grade=None,
            history=[],
        )
        
        # Save to class-level for persistence across instances
        self.__class__._class_state = self._state
        
        return self._build_observation(
            reward=None,
            done=False,
            feedback=[
                "Episode reset. Submit the full corrected module to receive a graded reward."
            ],
        )

    def step(
        self,
        action: CodeFixAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> CodeReviewObservation:
        if self._task is None:
            raise RuntimeError("Call reset() before step().")

        if self._state.step_count >= self._state.max_steps:
            return self._build_observation(
                reward=0.0,
                done=True,
                feedback=["The current episode has ended. Call reset() to start a new one."],
            )

        report = grade_task(
            self._task,
            action.candidate_code,
            timeout_s=timeout_s or self.default_timeout_s,
        )
        next_step = self._state.step_count + 1
        reward = round(report.score - self._state.previous_score, 4)
        done = report.score >= 1.0 or next_step >= self._state.max_steps
        reward_signal = RewardSignal(
            reward=reward,
            current_score=report.score,
            previous_score=self._state.previous_score,
            best_score=max(self._state.best_score, report.score),
            solved=report.score >= 1.0,
        )

        self._state.step_count = next_step
        self._state.current_code = action.candidate_code
        self._state.previous_score = report.score
        self._state.best_score = max(self._state.best_score, report.score)
        self._state.last_reward = reward
        self._state.last_reward_signal = reward_signal
        self._state.latest_grade = report
        self._state.history.append(
            AttemptRecord(
                step_index=next_step,
                score=report.score,
                reward=reward,
                summary=report.summary,
                failing_checks=[
                    check.name for check in report.checks if not check.passed
                ],
            )
        )

        feedback = list(report.feedback) or [report.summary]
        if done and report.score < 1.0:
            feedback.append("Step budget exhausted before reaching a perfect score.")
        elif done and report.score == 1.0:
            feedback.append("Task solved with a perfect score.")

        return self._build_observation(
            reward=reward,
            done=done,
            feedback=feedback,
        )

    @property
    def state(self) -> CodeReviewState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="CodeReviewEnvironment",
            description=(
                "An OpenEnv-compatible RL environment where agents repair buggy "
                "Python modules and receive dense reward from hidden unit checks."
            ),
            version="0.1.0",
            author="OpenEnv Hackathon",
            documentation_url="https://github.com/meta-pytorch/OpenEnv",
        )

    def _build_observation(
        self,
        *,
        reward: float | None,
        done: bool,
        feedback: list[str],
    ) -> CodeReviewObservation:
        if self._task is None:
            raise RuntimeError("No active task is loaded. Call reset() first.")

        latest = self._state.latest_grade
        return CodeReviewObservation(
            done=done,
            reward=reward,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            title=self._task.title,
            prompt=self._task.prompt,
            buggy_code=self._task.buggy_code,
            current_code=self._state.current_code,
            feedback=feedback,
            score=self._state.previous_score,
            best_score=self._state.best_score,
            tests_passed=latest.tests_passed if latest else 0,
            total_tests=latest.total_tests if latest else 0,
            remaining_steps=max(self._state.max_steps - self._state.step_count, 0),
            public_examples=self._task.public_examples,
            score_breakdown=latest.checks if latest else [],
            reward_signal=self._state.last_reward_signal,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "tags": self._task.tags,
            },
        )
