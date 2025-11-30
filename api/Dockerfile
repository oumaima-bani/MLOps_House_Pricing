# Stage 1: The Builder
# MUST be updated to a newer Python version
FROM python:3.11-slim as builder

# ... rest of builder stage ...
RUN pip install huggingface_hub
ARG HF_TOKEN
WORKDIR /model_cache
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='oumaouma/house-price-model', filename='model.onnx', local_dir='.', token='${HF_TOKEN}')"


# Stage 2: The Final Application Image
# MUST also be updated to the same newer Python version
FROM python:3.11-slim

# ... rest of final stage ...
WORKDIR /app
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt
COPY app.py .
COPY --from=builder /model_cache/model.onnx .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
