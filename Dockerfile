FROM python:3.9-slim

# Install only required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install gunicorn + gevent
RUN pip install --no-cache-dir gevent gunicorn

# Copy project files
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONMALLOC=malloc

CMD ["gunicorn", "--workers=2", "--timeout=120", "--log-level=debug", "--max-requests=1000", "--max-requests-jitter=50", "main:app"]