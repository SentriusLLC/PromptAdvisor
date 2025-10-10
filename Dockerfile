FROM python:3.11-bookworm

WORKDIR /app

# Install system dependencies
#RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
    libblas-dev liblapack-dev gfortran build-essential


# Copy requirements early for caching
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

ENV NLTK_DATA=/usr/local/share/nltk_data
RUN mkdir -p $NLTK_DATA && \
    python -m textblob.download_corpora lite && \
    python -m nltk.downloader -d $NLTK_DATA punkt punkt_tab averaged_perceptron_tagger wordnet omw-1.4

# Copy code
COPY . .
COPY prompt_advisor ./prompt_advisor

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /usr/local/share/nltk_data
USER appuser

EXPOSE 8000

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

CMD ["python", "-m", "uvicorn", "prompt_advisor.main:app", "--host", "0.0.0.0", "--port", "8000"]
