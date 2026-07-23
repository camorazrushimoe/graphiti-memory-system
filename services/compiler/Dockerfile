FROM python:3.12-slim

WORKDIR /app

# Install service dependencies
COPY services/compiler/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Dev-only deps (pytest) — separate layer/file on purpose, see
# requirements-dev.txt's docstring for rationale. Installed unconditionally
# (no build-arg toggle) since this is a single personal-scale image, not a
# prod/dev split — simplicity of "tests always runnable in the running
# container" wins over shaving a few MB off the image.
COPY services/compiler/requirements-dev.txt ./requirements-dev.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy shared schema + common packages, then service code
COPY schema/ ./schema/
COPY common/ ./common/
COPY services/compiler/ ./services/compiler/

ENV PYTHONPATH=/app

CMD ["python", "-m", "services.compiler.main"]
