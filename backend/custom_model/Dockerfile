FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip --timeout 1000
RUN pip install --no-cache-dir -r requirements.txt --timeout 1000

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

CMD ["uvicorn", "food_recognition_api:app", "--host", "0.0.0.0", "--port", "8001"]