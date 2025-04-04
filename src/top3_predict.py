import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/features_enhanced_final.csv")

# Drop rows with missing target
df = df.dropna(subset=["AvgLapTime"])

# Load model
model = joblib.load("models/lap_time_model.pkl")

# Predict
X = df.drop(columns=["AvgLapTime"])
predicted_times = model.predict(X)

# Add predictions to dataframe
df["PredictedLapTime"] = predicted_times

# Get Top 3 drivers with lowest predicted lap time
top3 = df.sort_values(by="PredictedLapTime").head(3)

# Print results
print("\n TOP 3 PREDICTED DRIVERS ")
print(top3[["DriverNumber", "PredictedLapTime"]])

# Plot - Bar chart
plt.figure(figsize=(8, 5))
plt.bar(top3["DriverNumber"].astype(str), top3["PredictedLapTime"], color="skyblue")
plt.xlabel("Driver Number")
plt.ylabel("Predicted Avg Lap Time")
plt.title("Top 3 Predicted Fastest Drivers")
plt.tight_layout()

# Save chart
plt.savefig("visuals/top3_predicted.png")
plt.show()
