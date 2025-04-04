import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load your CSV file
df = pd.read_csv("data/features_enhanced_final.csv")

# Target = AvgLapTime
X = df.drop(columns=["AvgLapTime"])
y = df["AvgLapTime"]

# Define numeric and categorical features
numeric = ["TotalLaps", "NumPitStops", "AvgPitDuration", "SafetyCarLaps", 
           "PitStopEfficiency", "LapTimeEfficiency"]
categorical = ["WeatherCondition"]

# Preprocessing + RandomForest model pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(), categorical)
])

pipeline = Pipeline([
    ("pre", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model
joblib.dump(pipeline, "models/lap_time_model.pkl")
print("Model trained and saved successfully!")
