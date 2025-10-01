# src/train_model.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# 1. Load dataset
DATA_PATH = r"C:\Users\N B Kundan Setty\Documents\CreditCardFraud\data\creditcard.csv"
df = pd.read_csv(DATA_PATH)

# 2. Split features and labels
X = df.drop("Class", axis=1)
y = df["Class"]

# 3. Train-test split (stratify to keep class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Handle class imbalance
# Compute scale_pos_weight (majority/minority ratio)
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

# 5. Initialize XGBoost classifier
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

# 6. Train model
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# 8. Save trained model
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "xgb_fraud_model.pkl"))

print(f"\nModel saved to {os.path.join(MODEL_DIR, 'xgb_fraud_model.pkl')}")