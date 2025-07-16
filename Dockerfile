# Dockerfile for H&M Recommendation API
FROM python:3.9-slim

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

# Copy source code
COPY src/ ./src/
COPY scripts/serve.py ./scripts/
COPY configs/ ./configs/

# Create directories for data and models
RUN mkdir -p data/processed experiments/checkpoints

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data
ENV CHECKPOINT_DIR=/app/experiments/checkpoints

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]