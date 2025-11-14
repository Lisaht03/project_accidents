# ============================================
# Dockerfile para FastAPI - local + GCP
# ============================================

# Base Python slim
FROM python:3.12.9-slim

#Copy code and essential files.
COPY models models
COPY project_accidents_package project_accidents_package
COPY requirements.txt requirements.txt
COPY setup.py setup.py

#Update pip and install dependencies.
RUN pip install --upgrade pip
RUN pip install -e .

ENV PORT=8080

CMD ["python", "project_accidents_package/api_file.py"]
#Run container locally
# CMD uvicorn project_accidents_package.api_file:app --reload --host 0.0.0.0

#Run container GCP
#CMD uvicorn project_accidents_package.api_file:app --reload --host 0.0.0.0 --port $PORT
