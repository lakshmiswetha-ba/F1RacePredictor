import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error

def train_model(input_path, output_model_path):
    df = pd.read_csv(input_path)
    df = df.dropna(subset=["RacePosition"])

    X = df.drop(columns=["RacePosition", "Driver", "Team"])
    y = df["RacePosition"]

    numeric_features = ["TotalLaps", "AvgLapTime", "NumPitStops", "AvgPitDuration",
                        "SafetyCarLaps", "PitStopEfficiency", "LapTimeEfficiency"]
    categorical_features = ["WeatherCondition"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(), categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae:.2f}")

    joblib.dump(pipeline, output_model_path)
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    train_model(
        input_path="data/features_with_position.csv",
        output_model_path="models/position_model.pkl"
    )
