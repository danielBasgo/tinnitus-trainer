# Dockerfile (CPU-Only)

# 1. Base Image: Use an official, slim Python image.
# 'python:3.11-slim' is a good choice as it's smaller than the full version.
FROM python:3.11-slim-bullseye

# Ensure all security updates are installed
RUN apt-get update && apt-get upgrade -y && apt-get clean

# 2. Set the working directory inside the container.
# This is like running 'cd /app' and all subsequent commands will run here.
WORKDIR /app

# 3. Copy ONLY the requirements file first.
# This is Docker's brilliant caching trick. If this file doesn't change,
# Docker reuses the already installed packages from the last build,
# making future builds much faster.
COPY requirements.txt .

# 4. Install system dependencies and Python dependencies.
# Install build-essential and other libraries required for scientific packages.
RUN apt-get update && \
	apt-get install -y --no-install-recommends build-essential gcc libffi-dev libssl-dev && \
	rm -rf /var/lib/apt/lists/*

# --no-cache-dir makes the image slightly smaller.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project code into the container's working directory.
# This includes train.py, your 'processed_data' folder, etc.
COPY . .

# 6. Define the command that will run when the container starts.
# This will execute 'python train.py' inside the container.
CMD ["python", "train.py"]