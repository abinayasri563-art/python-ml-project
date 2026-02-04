import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =======================
# Load data
# =======================
df = pd.read_csv("house price/data.csv")

# DROP string columns (VERY IMPORTANT)
df = df.drop(columns=["date", "street", "city", "statezip", "country"])

# =======================
# Features & target
# =======================
X = df.drop("price", axis=1)
y = df["price"]

# =======================
# Train-test split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =======================
# Train model
# =======================
model = LinearRegression()
model.fit(X_train, y_train)

# =======================
# Predict
# =======================
y_pred = model.predict(X_test)

# =======================
# Evaluation
# =======================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")

# =======================
# Plot
# =======================
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("House Price Prediction")
plt.show(block=True)
