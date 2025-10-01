from fastapi import FastAPI
from pydantic import BaseModel
from kafka import KafkaProducer
import json
import joblib
import numpy as np

# Load trained model
MODEL_PATH = "src/models/xgb_fraud_model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Credit Card Fraud Detection API")

# Kafka producer for optional manual sending
KAFKA_TOPIC = "raw-transactions"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Pydantic model
class Transaction(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "Credit Card Fraud Detection API running!"}

@app.post("/send/")
def send_transaction(transaction: Transaction):
    """Send transaction to Kafka"""
    producer.send(KAFKA_TOPIC, value=transaction.dict())
    producer.flush()
    return {"status": "Transaction sent to Kafka"}