FROM python:3.9-slim

WORKDIR /app

# Copy root requirements.txt
COPY requirements_api.txt ./requirements.txt
RUN pip install --default-timeout=600 --no-cache-dir -r requirements.txt

# Copy FastAPI code from api folder
COPY api/FastAPI.py .

EXPOSE 8000

CMD ["uvicorn", "FastAPI:app", "--host", "0.0.0.0", "--port", "8000"]
