import pandas as pd
import os


URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


os.makedirs("data", exist_ok=True)


df = pd.read_csv(URL)
df.to_csv("data/customer_churn.csv", index=False)

print(" Data downloaded and saved to data/customer_churn.csv")
