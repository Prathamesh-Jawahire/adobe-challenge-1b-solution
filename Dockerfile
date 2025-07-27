FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    --retries 10 --timeout 600 \
    -r requirements.txt

# Create models directory
RUN mkdir -p models

# Copy and run model download script
COPY download_models.py .
RUN python3 download_models.py && rm download_models.py

# Copy application files
COPY combined.py round_1A.py round_1B.py challenge1b_input.json ./
COPY entrypoint.sh .

RUN mkdir -p input && chmod +x entrypoint.sh

# Set offline environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

ENTRYPOINT ["./entrypoint.sh"]
