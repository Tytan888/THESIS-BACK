# ==========================================
# STAGE 1: The Model Downloader
# ==========================================
FROM python:3.13-slim AS model-downloader

RUN pip install sentence-transformers
ENV HF_HOME=/models

RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token) \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/LaBSE')"

# ==========================================
# STAGE 2: The Builder
# ==========================================
FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install/lib --no-cache-dir -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    PYTHONPATH=/install/lib/lib/python3.13/site-packages python -m spacy download en_core_web_sm && \
    cp -r /usr/local/lib/python3.13/site-packages/en_core_web_sm* /install/lib/lib/python3.13/site-packages/
    
# ==========================================
# STAGE 3: The Final Lean Image
# ==========================================
FROM python:3.13-slim

# 1. Install OS Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Tell Hugging Face to look in our custom directory
ENV HF_HOME=/models

# 3. Copy heavy static assets
COPY --from=model-downloader /models /models
COPY jamdict.db /root/.jamdict/data/jamdict.db

# 4. Copy pip packages
COPY --from=builder /install/lib /usr/local

# 5. Copy and install local package
COPY awesome-align /app/awesome-align
RUN pip install -e /app/awesome-align

# 6. Copy the rest of the fast-changing application code
COPY . .

# 7. Expose ports and set variables
EXPOSE 8080
ENV PYTHONPATH="/app/app:${PYTHONPATH}"

# 8. Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]