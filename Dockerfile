# Dockerfile for XFL-RPiLab Client
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy client code
COPY client/ ./client/
COPY server/ ./server/
COPY dashboard/ ./dashboard/
COPY config/ ./config/
COPY run_dashboard.py .

# Create data and logs directories
RUN mkdir -p /app/data /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (will be overridden by docker-compose)
CMD ["python", "-m", "client.run_client_standalone"]