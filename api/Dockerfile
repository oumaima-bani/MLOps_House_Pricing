# Stage 1: The Builder
# This stage downloads the model from Hugging Face
FROM python:3.9-slim as builder

# Install only the dependency needed for downloading
RUN pip install huggingface_hub

# Set your Hugging Face token as a build argument
# IMPORTANT: You will need to pass this when building the image
ARG HF_TOKEN

# Create a directory for the model
WORKDIR /app/model
# Download the model file into this directory
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='oumaouma/house-price-model', filename='model.onnx', local_dir='.', token='${HF_TOKEN}')"


# Stage 2: The Final Application Image
# This is the lean image that will be deployed
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install them
COPY requirements_api.txt .
# Using --no-cache-dir is a good practice for smaller images
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy your application code
COPY app.py .

# Copy the downloaded model from the 'builder' stage into the final image
COPY --from=builder /app/model/model.onnx .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the app
# The model is now loaded from a local file, not downloaded
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]