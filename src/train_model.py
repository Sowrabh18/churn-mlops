import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Load data
df = pd.read_csv("data/customer_churn.csv")

# Preprocess: drop customerID (not useful)
df.drop("customerID", axis=1, inplace=True)

# Convert categorical to numerical
df = pd.get_dummies(df)

# Split
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# MLflow Tracking
mlflow.set_experiment("churn-prediction")

with mlflow.start_run():
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("max_iter", 1000)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)

    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="logistic_churn_model")

print("✅ Model trained, saved, and logged with MLflow")
