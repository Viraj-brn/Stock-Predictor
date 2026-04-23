FROM python:3.11-slim

WORKDIR /app

# Install deps first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port Render expects
EXPOSE 5000

# Optimize PyTorch memory footprint for Render free tier
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Run with gunicorn (1 worker to stay within free tier RAM)
CMD gunicorn -w 1 -b 0.0.0.0:${PORT:-5000} --timeout 120 backend.app:app
