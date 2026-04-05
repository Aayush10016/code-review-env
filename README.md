---
title: CodeReviewEnv
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - code-review
  - python
  - reinforcement-learning
  - real-world
license: mit
---

# CodeReviewEnv

CodeReviewEnv is a real-world OpenEnv environment for training and evaluating agents that fix buggy Python code. Each episode presents a broken module, asks the agent to submit a full corrected replacement, and returns shaped reward from deterministic hidden graders.

## Why this environment

Code review and bug fixing are practical, high-value workflows that software teams perform every day. This environment models a realistic version of that task with:

- concrete debugging objectives
- hidden deterministic graders
- partial-credit reward shaping
- multiple difficulty levels
- Docker-first deployment for reproducible evaluation

## Environment design

### API

- `reset(seed, task_id, difficulty)` selects a task and returns the initial observation
- `step(action)` grades a candidate fix and returns the next observation plus reward
- `state()` exposes the current episode snapshot, including history and best score

### Action space

`CodeFixAction`

- `candidate_code: str`
  The full Python module the agent wants to submit as the fix.
- `notes: Optional[str]`
  Optional reasoning text from the agent.

### Observation space

`CodeReviewObservation`

- task metadata: `task_id`, `difficulty`, `title`, `prompt`
- source context: `buggy_code`, `current_code`
- grading feedback: `feedback`, `score_breakdown`
- progress: `score`, `best_score`, `tests_passed`, `total_tests`, `remaining_steps`

### Reward space

`RewardSignal`

- `reward`
- `current_score`
- `previous_score`
- `best_score`
- `solved`

### State space

`CodeReviewState`

- active episode metadata
- current code submission
- latest grade report
- previous and best scores
- full attempt history

### Client

An OpenEnv-style reusable client is included at [client.py](C:/Users/Abhinav%20Jain/Downloads/New%20folder%20(5)/client.py) and [code_review_env/client.py](C:/Users/Abhinav%20Jain/Downloads/New%20folder%20(5)/code_review_env/client.py). It uses the official `EnvClient` when `openenv-core` is installed and falls back to plain HTTP otherwise.

### Reward design

Each task has weighted hidden checks that sum to `1.0`. The grader returns a normalized score in `[0.0, 1.0]`, and the environment reward is:

```text
reward = current_score - previous_score
```

This gives dense partial-progress signal across the trajectory instead of only terminal binary reward.

## Tasks

### 1. Easy: `easy_dedupe`

Fix an order-preserving deduplication helper.

Target behavior:

- remove duplicates
- preserve first occurrence order
- return a list

### 2. Medium: `medium_merge_intervals`

Fix an interval-merging utility used by scheduling systems.

Target behavior:

- sort by start time
- merge overlapping intervals
- merge touching intervals
- return tuples

### 3. Hard: `hard_lru_cache`

Fix an LRU cache implementation.

Target behavior:

- `get()` refreshes recency
- `put()` updates existing keys correctly
- capacity is enforced
- eviction removes the least recently used item

## Project structure

```text
.
|-- app.py
|-- client.py
|-- Dockerfile
|-- inference.py
|-- openenv.yaml
|-- pyproject.toml
|-- requirements.txt
|-- README.md
|-- server/
|   |-- __init__.py
|   |-- app.py
|   |-- code_review_environment.py
|   |-- Dockerfile
|   `-- requirements.txt
`-- code_review_env/
    |-- __init__.py
    |-- _grade_worker.py
    |-- compat.py
    |-- client.py
    |-- environment.py
    |-- graders.py
    |-- inference.py
    |-- models.py
    |-- tasks.py
    `-- server/
        |-- app.py
        |-- code_review_environment.py
        |-- Dockerfile
        `-- requirements.txt
```

## Local setup

Install:

```bash
pip install -r requirements.txt
pip install .
```

Run the environment server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Inspect available tasks:

```bash
curl http://127.0.0.1:7860/tasks
```

Run the validation endpoint:

```bash
curl http://127.0.0.1:7860/validate
```

Run OpenEnv validation:

```bash
openenv validate --verbose
```

## Docker

Build:

```bash
docker build -t code-review-env:latest .
```

Or using OpenEnv:

```bash
openenv build
```

Run:

```bash
docker run --rm -p 7860:7860 code-review-env:latest
```

## Baseline inference

The required baseline script is the root-level `inference.py`. It uses the OpenAI Python client and emits structured stdout logs with `[START]`, `[STEP]`, and `[END]`.

If `ENV_BASE_URL` points to `localhost` or `127.0.0.1` and no server is already running, the script automatically starts the local environment before scoring tasks.

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional:

- `OPENAI_API_KEY`
  Used if you want to override `HF_TOKEN` as the OpenAI client API key.
- `ENV_BASE_URL`
  Defaults to `http://127.0.0.1:7860`

Run:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="openai/gpt-4.1-mini"
export HF_TOKEN="your-token"
python inference.py
```

## Baseline scores

This sandbox does not currently expose a Python interpreter, so final reproducible baseline numbers still need to be generated by running the root inference script in a normal Python environment before submission.

Recommended final step before submission:

1. Run `python inference.py`
2. Capture the final score for each task from the `[END]` lines
3. Paste those three scores and their average into this section

## Submission checklist

- root-level `inference.py`
- root-level `openenv.yaml`
- root-level `client.py`
- Dockerfile for Hugging Face Spaces
- typed Pydantic action, observation, reward, state, and grading models
- `reset()`, `step()`, `state()`
- 3 deterministic tasks with easy, medium, hard progression
- shaped rewards with partial progress signals
- `/tasks` and `/validate` helper endpoints

## Sources

- [OpenEnv docs](https://meta-pytorch.org/OpenEnv/)
- [OpenEnv GitHub repository](https://github.com/meta-pytorch/OpenEnv)
- [OpenEnv environment interface](https://raw.githubusercontent.com/meta-pytorch/OpenEnv/main/src/openenv/core/env_server/interfaces.py)
- [OpenEnv HTTP server implementation](https://raw.githubusercontent.com/meta-pytorch/OpenEnv/main/src/openenv/core/env_server/http_server.py)
