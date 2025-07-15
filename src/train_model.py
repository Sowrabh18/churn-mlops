import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import yaml

# 🛠️ Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

test_size = params["split"]["test_size"]
random_state = params["split"]["random_state"]
C = params["train"]["C"]
max_iter = params["train"]["max_iter"]

# 📊 Load and preprocess data
df = pd.read_csv("data/customer_churn.csv")
df.drop("customerID", axis=1, inplace=True)
df = pd.get_dummies(df)
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# 🧠 Train model
model = LogisticRegression(C=C, max_iter=max_iter)
model.fit(X_train, y_train)

# 📈 Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 💾 Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

# 🚀 MLflow Logging
mlflow.set_tracking_uri("file:mlruns")  # Let MLflow manage this folder
mlflow.set_experiment("churn-prediction")  # Auto-creates experiment and meta.yaml

run_name = os.getenv("GITHUB_RUN_ID", "local-run")

with mlflow.start_run(run_name=run_name):
    mlflow.set_tag("ci", "github-actions")
    mlflow.set_tag("workflow", os.getenv("GITHUB_WORKFLOW", "unknown"))
    mlflow.set_tag("job", os.getenv("GITHUB_JOB", "unknown"))

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("C", C)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="logistic_churn_model",
        input_example=X_test.iloc[:1]
    )

print("✅ MLflow run logged successfully.")
