import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load trained model and data
model = joblib.load("models/lap_time_model.pkl")
df = pd.read_csv("data/features_enhanced_final.csv")

X = df.drop(columns=["AvgLapTime"])
y = df["AvgLapTime"]

# Predict lap times
y_pred = model.predict(X)

# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)

# Plot actual vs predicted
plt.figure(figsize=(8, 5))
plt.plot(y.values, label="Actual", marker='o')
plt.plot(y_pred, label="Predicted", marker='x')
plt.title("Predicted vs Actual Avg Lap Time")
plt.xlabel("Driver Index")
plt.ylabel("Lap Time (s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("visuals/lap_time_plot.png")  # Save chart
plt.show()

print(f" RMSE: {rmse:.4f}, MAE: {mae:.4f}")
