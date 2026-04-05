FROM python:3.11-slim

ENV PORT=7860
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY code_review_env /app/code_review_env

RUN pip install --upgrade pip \
    && pip install .

EXPOSE 7860

CMD ["uvicorn", "code_review_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
