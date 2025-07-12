import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Load data
df = pd.read_csv("data/customer_churn.csv")

# Drop customerID (non-numeric and not useful for prediction)
df.drop("customerID", axis=1, inplace=True)

# Convert target column 'Churn' to binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Fill missing values (if any)
df = df.fillna(0)

# Convert categorical variables using one-hot encoding
df = pd.get_dummies(df)

# Split into features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Model saved to models/model.pkl")
