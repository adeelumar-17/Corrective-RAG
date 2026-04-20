FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
