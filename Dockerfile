FROM python:3.11-slim-bullseye

# Install only necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libffi-dev libssl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Optionally, run pip-audit to check for Python package vulnerabilities
# RUN pip install pip-audit && pip-audit

# Use a non-root user for extra security (optional)
# RUN useradd -m appuser && chown -R appuser /app
# USER appuser