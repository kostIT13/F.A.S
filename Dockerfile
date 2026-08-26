FROM python:3.12-slim

WORKDIR /app

# Установка системных зависимостей для FAISS
RUN apt-get update && apt-get install -y \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка uv
RUN pip install uv

# Копирование файлов проекта
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Установка зависимостей
RUN uv pip install --system -e .

# Создание директории для коллекций
RUN mkdir -p /app/collections

EXPOSE 8000

# Запуск: src.main:app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]