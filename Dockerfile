FROM python:3.11-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt
COPY src ./src
COPY api ./api
COPY models/deploy ./models/deploy
COPY configs ./configs
ENV DEPLOY_DIR=/app/models/deploy CLS_CKPT=/app/models/deploy/classifier_best.pt PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
