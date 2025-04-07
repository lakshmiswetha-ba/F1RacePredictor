import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def train_laptime_model(input_path: str, model_path: str):
    """
    Train a RandomForest model to predict average lap time based on features.
    
    Args:
        input_path (str): Path to the features CSV.
        model_path (str): Path to save the trained model.
    """
    # Load data
    df = pd.read_csv(input_path)
    X = df.drop(columns=["AvgLapTime"])
    y = df["AvgLapTime"]

    # Define feature types
    numeric = ["TotalLaps", "NumPitStops", "AvgPitDuration", 
               "SafetyCarLaps", "PitStopEfficiency", "LapTimeEfficiency"]
    categorical = ["WeatherCondition"]

    # Build preprocessing + model pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate quickly
    score = pipeline.score(X_test, y_test)
    print(f"Model R² Score on Test Set: {score:.4f}")

    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")


# Run standalone
if __name__ == "__main__":
    train_laptime_model(
        input_path="data/features_enhanced_final.csv",
        model_path="models/lap_time_model.pkl"
    )
