FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY api ./api
COPY apps ./apps
COPY docs ./docs
COPY data ./data

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .

EXPOSE 8000 8501

CMD ["python", "-m", "movie_recommender.cli.main", "serve-api"]

