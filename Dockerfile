# ============================================
# Optimized Dockerfile for GCP Cloud Run (Python 3.12)
# ============================================

# 1. Base Image
# We use the 'slim' version to reduce the final image size and deployment time.
# It matches the Python version used during model training.
FROM python:3.12-slim

# 2. Set Working Directory
# All subsequent commands will be run from inside /app within the container.
WORKDIR /app

# 3. Install System Dependencies
# 'build-essential' includes C++ compilers required by libraries like Pandas and Scikit-learn.
# We clean up the apt cache ('rm -rf') to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# 4. Install Python Dependencies
# We copy requirements.txt first to leverage Docker's layer caching.
# (If requirements haven't changed, Docker skips this step on rebuilds).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy Project Structure
# Copy the trained models folder into the container (/app/models)
COPY models/ ./models/

# Copy the source code package into the container (/app/project_accidents_package)
COPY project_accidents_package/ ./project_accidents_package/

# 6. Environment Variables
# GCP Cloud Run expects the application to listen on port 8080 by default.
ENV PORT=8080

# 7. Execution Command
# This command starts the Uvicorn server when the container launches.
# Syntax: 'package_name.file_name:app_instance_name'
# --host 0.0.0.0: Required to make the container accessible from outside.
CMD uvicorn project_accidents_package.api_file:app --host 0.0.0.0 --port $PORT
