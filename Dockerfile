FROM python:3.10-slim

# Установка системных зависимостей для lightgbm, xgboost и других библиотек
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем и устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY main.py .

# Создаем папку для результатов
RUN mkdir -p /app/results

# Команда запуска
CMD ["python", "main.py"]