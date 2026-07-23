FROM python:3.12-slim

WORKDIR /app

# Install service dependencies
COPY services/compiler/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared schema + common packages, then service code
COPY schema/ ./schema/
COPY common/ ./common/
COPY services/compiler/ ./services/compiler/

ENV PYTHONPATH=/app

CMD ["python", "-m", "services.compiler.main"]
