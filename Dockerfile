FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy app code
COPY . .

# Cache directory — persists bhavcopy data across restarts
RUN mkdir -p /cache/nse_bhav_days /cache/nse_ohlcv_pkl

# Point the app's cache dirs to /cache (mounted as a volume on EC2)
ENV BHAV_DIR=/cache/nse_bhav_days
ENV OHLCV_DIR=/cache/nse_ohlcv_pkl

EXPOSE 5050

# Gunicorn: 2 workers, 120s timeout (scans take time)
CMD ["gunicorn", "--workers=2", "--timeout=120", "--bind=0.0.0.0:5050", "app:app"]
