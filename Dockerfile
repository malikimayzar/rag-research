FROM python:3.12-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install only the runtime package needed by the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependencies required by the API runtime.
COPY requirements/runtime.txt /app/requirements/

# Install the CPU-only PyTorch wheel once. The base requirements must not list torch again.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --timeout 120 \
    --retries 2 \
    torch

# Keep dependency installation separate so application code changes do not
# invalidate the dependency layer.
RUN pip install --no-cache-dir -r requirements/runtime.txt

# Copy application code
COPY src/ /app/src/
COPY data/processed/ /app/data/processed/

# Create directory for runtime data
RUN mkdir -p /app/data/cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]