FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN uv pip install --system -e .

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]