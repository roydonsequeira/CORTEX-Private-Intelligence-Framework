FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY cortex.yaml ./

RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENTRYPOINT ["uvicorn", "cortex.api.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
