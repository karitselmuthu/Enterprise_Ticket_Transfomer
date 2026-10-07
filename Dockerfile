FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY api/ api/
COPY data/samples/tickets.csv data/samples/tickets.csv
COPY configs/sample_split.json configs/sample_split.json
RUN python -m src.training.train --data data/samples/tickets.csv --split configs/sample_split.json --model models/v1.json

ENV TICKET_MODEL_PATH=models/v1.json
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
