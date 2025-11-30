# Stage 1: The Builder
FROM python:3.10-slim as builder

RUN pip install huggingface_hub
ARG HF_TOKEN
WORKDIR /model_cache
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='oumaouma/house-price-model', filename='model.onnx', local_dir='.', token='${HF_TOKEN}')"


# Stage 2: The Final Application
FROM python:3.10-slim

WORKDIR /app

# === DIAGNOSTIC STEP ===
# Let's try to install onnxruntime-cpu all by itself first.
# If this command fails, the problem is with Render's environment.
RUN pip install onnxruntime-cpu

# Now, copy the requirements file and install the rest.
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy the rest of the application
COPY app.py .
COPY --from=builder /model_cache/model.onnx .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
