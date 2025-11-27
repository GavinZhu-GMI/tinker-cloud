# Dockerfile for kgateway AI Extension with Miles Training API support
# Based on miles image which has Ray 2.52.0, torch, transformers pre-installed

FROM us-west1-docker.pkg.dev/devv-404803/gmi-test-repo/miles_dev-202511120a:latest

LABEL maintainer="kgateway-dev"
LABEL description="kgateway AI Extension with Miles Training API support"
LABEL version="1.0"

ARG PYTHON_DIR

# Install additional kgateway dependencies (ext_proc and guardrails)
WORKDIR /tmp
COPY $PYTHON_DIR/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Pre-download tiktoken encoding model
ENV TIKTOKEN_CACHE_DIR="/opt/tiktoken"
RUN python3 -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" || true

ENV TLDEXTRACT_CACHE="/opt/tldextract"
RUN python3 -m tldextract --update || true

# Copy kgateway AI extension code
WORKDIR /app
COPY $PYTHON_DIR/ai_extension /app/ai_extension

# Copy test datasets for RL training validation
COPY $PYTHON_DIR/ai_extension/training/test_data /app/test_data

# Create legacy dummy.jsonl symlink for backward compatibility
RUN ln -s /app/test_data/gsm8k_rl.jsonl /tmp/dummy.jsonl || true

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/ai_extension:/root/Megatron-LM:/root/miles" \
    TLDEXTRACT_CACHE="/opt/tldextract" \
    TIKTOKEN_CACHE_DIR="/opt/tiktoken" \
    RUN_MODE="both" \
    TRAINING_PORT="8000" \
    LOG_LEVEL="INFO"

# Expose ports
# 8000 - Training API (Tinker API)
# 9092 - Prometheus metrics (from ext_proc)
EXPOSE 8000 9092

# Run unified server (both ext_proc and training)
CMD ["python3", "-m", "ai_extension.main_unified"]
