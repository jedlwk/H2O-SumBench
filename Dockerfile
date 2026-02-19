FROM python:3.10-slim

WORKDIR /app

# System deps for pyemd C extension
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLP models during build
RUN python -m spacy download en_core_web_sm && \
    python -c "import nltk; nltk.download('punkt_tab')"

# Copy application source
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
