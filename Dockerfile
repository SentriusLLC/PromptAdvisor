FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY prompt_validator ./prompt_validator

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Environment variables (can be overridden by Kubernetes ConfigMap)
ENV HOST=0.0.0.0
ENV PORT=8000
ENV ATPL_SCHEMA_URL=https://raw.githubusercontent.com/SentriusLLC/atpl/main/atpl.schema.json
ENV LLM_ENDPOINT=""
ENV LLM_API_KEY=""
ENV LLM_MODEL=gpt-4
ENV LLM_ENABLED=false
ENV WEIGHT_PURPOSE=15
ENV WEIGHT_SAFETY=30
ENV WEIGHT_COMPLIANCE=25
ENV WEIGHT_PROVENANCE=15
ENV WEIGHT_AUTONOMY=15

# Run the application
CMD ["python", "-m", "uvicorn", "prompt_validator.main:app", "--host", "0.0.0.0", "--port", "8000"]
