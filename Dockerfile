FROM python:3.9-slim-buster

# Install system packages including tesseract and libGL
RUN apt-get update && \
    apt-get -qq -y install tesseract-ocr libtesseract-dev libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# Adjust this if your entrypoint file is named differently
CMD ["gunicorn", "main:app"]