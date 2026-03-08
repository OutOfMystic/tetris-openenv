FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force rebuild v2
COPY src/ src/
COPY README.md README.md

ENV ENABLE_WEB_INTERFACE=true

EXPOSE 8000

RUN python -c "from src.tetris_env.server.app import app; print('Import OK, routes:', [r.path for r in app.routes])"

CMD ["uvicorn", "src.tetris_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
