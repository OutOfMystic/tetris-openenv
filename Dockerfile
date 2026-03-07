FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY README.md README.md

ENV ENABLE_WEB_INTERFACE=true

EXPOSE 7860

CMD ["uvicorn", "src.tetris_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
