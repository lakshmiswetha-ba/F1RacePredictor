import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv("data/features_enhanced_final.csv")

# Load the trained model
model = joblib.load("models/lap_time_model.pkl")

# Separate features and target
X = df.drop(columns=["AvgLapTime"])
y_true = df["AvgLapTime"]

# Predict lap times
y_pred = model.predict(X)
df["PredictedLapTime"] = y_pred

# Get top 3 predicted drivers with lowest predicted lap time
top3_predicted = df.sort_values("PredictedLapTime").head(3)["DriverNumber"].tolist()
print("Top 3 Predicted Driver Numbers:", top3_predicted)

# Get actual top 3 drivers with lowest actual lap time
top3_actual = df.sort_values("AvgLapTime").head(3)["DriverNumber"].tolist()
print("Top 3 Actual Driver Numbers:", top3_actual)

# Create bar chart comparing predicted vs actual lap times
plt.figure(figsize=(8, 5))
labels = ["1st", "2nd", "3rd"]

# ----------------------------------------------
# Feature Importance Chart (if model supports it)
# ----------------------------------------------
trained_model = model.named_steps["model"]      

if hasattr(trained_model, "feature_importances_"):
    preprocessor = model.named_steps["pre"]
    categorical_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out()
    numerical_feature_names = preprocessor.transformers_[0][2]
    feature_names = np.concatenate([numerical_feature_names, categorical_feature_names])
    importance_scores = trained_model.feature_importances_
    if len(feature_names) == len(importance_scores):
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance_scores
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Feature"], importance_df["Importance"])
        plt.xlabel("Importance")
        plt.title("Feature Importance from Trained Model")
        plt.tight_layout()
        plt.savefig("visuals/feature_importance.png")
        print("Feature importance chart saved to visuals/feature_importance.png")
        plt.show()
    else:
        print("Feature count doesn't match importance values. Skipping the plot.")
else:
    print("This model does not support feature importance.")
