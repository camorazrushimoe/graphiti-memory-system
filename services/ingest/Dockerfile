FROM python:3.12-slim

WORKDIR /app

# Install service dependencies
COPY services/ingest/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared schema + common packages, then service code
COPY schema/ ./schema/
COPY common/ ./common/
COPY services/ingest/ ./services/ingest/

ENV PYTHONPATH=/app
WORKDIR /app/services/ingest

EXPOSE 8100

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
