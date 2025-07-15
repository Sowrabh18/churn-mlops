import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import yaml
import logging

# Enable MLflow debug logging
logging.getLogger("mlflow").setLevel(logging.DEBUG)

# Debug file paths and environment
print("Current working directory:", os.getcwd())
print("params.yaml exists:", os.path.exists("params.yaml"))
print("customer_churn.csv exists:", os.path.exists("data/customer_churn.csv"))
print("MLflow version:", mlflow.__version__)

# MLflow setup
mlflow.set_tracking_uri("file:./mlruns")
print("MLflow tracking URI:", mlflow.get_tracking_uri())
mlflow.set_experiment("churn-prediction")

# Load parameters
if not os.path.exists("params.yaml"):
    raise FileNotFoundError("params.yaml not found")
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

test_size = params["split"]["test_size"]
random_state = params["split"]["random_state"]
C = params["train"]["C"]
max_iter = params["train"]["max_iter"]

# Load and preprocess data
if not os.path.exists("data/customer_churn.csv"):
    raise FileNotFoundError("customer_churn.csv not found")
df = pd.read_csv("data/customer_churn.csv")
df.drop("customerID", axis=1, inplace=True)
df = pd.get_dummies(df)

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train model
model = LogisticRegression(C=C, max_iter=max_iter)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Log everything
try:
    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        print("Model saved at models/model.pkl")

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact("models/model.pkl", artifact_path="model")
        print("MLflow logging completed")
except Exception as e:
    print(f"MLflow logging failed: {e}")
    raise

print("✅ Model trained, saved, and logged with MLflow")


