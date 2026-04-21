# Dockerfile for XFL-RPiLab Server
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install build dependencies for numpy
RUN pip install --no-cache-dir setuptools wheel

# Copy requirements
COPY requirements_server.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_server.txt

# Copy server code
COPY server/ ./server/
COPY dashboard/ ./dashboard/
COPY config/ ./config/


# Create data and logs directories
RUN mkdir -p /app/data /app/logs /app/results

# Set environment variables
ENV PYTHONUNBUFFERED=1

# PyTorch memory optimization settings
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Default command
CMD ["python", "-m", "server.run_server_standalone"]
