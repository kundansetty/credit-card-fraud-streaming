from kafka import KafkaProducer
import json
import time
import random

KAFKA_TOPIC = "raw-transactions-new"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_transaction():
    features = [random.uniform(-2, 2) for _ in range(30)]
    features[-1] = random.uniform(1, 500)  # Amount
    return {"features": features}

if __name__ == "__main__":
    repeat = 0
    while repeat <= 10:
        txn = generate_transaction()
        producer.send(KAFKA_TOPIC, value=txn)
        producer.flush()
        print(f"📤 Sent transaction: {txn}")
        time.sleep(1)
        repeat += 1