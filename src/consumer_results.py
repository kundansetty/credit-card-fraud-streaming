from kafka import KafkaConsumer
import json
import joblib
import numpy as np

# Load model
MODEL_PATH = "models/xgb_fraud_model.pkl"
model = joblib.load(MODEL_PATH)

KAFKA_TOPIC = "raw-transactions-new"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    auto_offset_reset='latest',
    group_id='fraud-consumer-group',
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

print("🚀 Kafka consumer started. Listening for transactions...")

try:
    for msg in consumer:
        txn = msg.value
        features = np.array(txn["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        result = {
            "fraud_prediction": int(prediction),
            "fraud_probability": float(probability),
            "transaction": txn
        }
        print(f"📥 Prediction: {result}")
except KeyboardInterrupt:
    print("🛑 Consumer stopped by user")
finally:
    consumer.close()