# FROM python:3.9-slim-buster

# # Install system packages including tesseract and libGL
# RUN apt-get update && \
#     apt-get -qq -y install tesseract-ocr libtesseract-dev libgl1 && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# COPY requirements.txt requirements.txt
# RUN pip3 install --no-cache-dir -r requirements.txt

# COPY . .

# # Adjust this if your entrypoint file is named differently
# CMD ["gunicorn", "main:app"]
FROM python:3.9-slim-buster

# Install system packages including tesseract and libGL
RUN apt-get update && \
    apt-get -qq -y install tesseract-ocr libtesseract-dev libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements first (for better caching)
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Make sure gevent is installed
RUN pip3 install --no-cache-dir gevent gunicorn

# Copy application code
COPY . .

# Set environment variable to reduce Python buffer usage
ENV PYTHONUNBUFFERED=1

# Memory optimization: Limit Python's memory usage
ENV PYTHONMALLOC=malloc

# Configure Gunicorn with gevent workers for better memory handling
CMD ["gunicorn", "--worker-class=gevent", "--worker-connections=1000", "--workers=3", "--timeout=120", "--log-level=info", "--preload", "main:app"]