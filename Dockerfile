FROM python:3.10.6-slim

WORKDIR /app

COPY models ./models
COPY data ./data
COPY project_accidents_package ./project_accidents_package
COPY requirements.txt .
COPY setup.py .

RUN pip install --upgrade pip
RUN pip install -e .

# As we need to deploy it on GCP using the Cloud Run, here is the CMD line to ensure the service listens on the correct port
CMD uvicorn road_accident_package.api_file:app --host 0.0.0.0 --port $$PORT
