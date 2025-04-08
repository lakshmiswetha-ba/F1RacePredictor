import pandas as pd
import joblib

def predict_race_positions(input_csv, model_path):
    # Load the feature data with actual positions
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["RacePosition"])

    # Separate features from the target
    X = df.drop(columns=["RacePosition", "Driver", "Team"])
    y_true = df["RacePosition"]

    # Load the trained model
    model = joblib.load(model_path)

    # Predict race positions
    predicted_positions = model.predict(X)

    # Round predictions to nearest integer
    df["PredictedPosition"] = predicted_positions.round()

    # Keep useful columns for review
    output = df[["DriverNumber", "Driver", "Team", "RacePosition", "PredictedPosition"]]
    output = output.sort_values("PredictedPosition")

    # Print top 10 predictions
    print(output.head(10))

    # Save predictions to CSV
    output.to_csv("outputs/predicted_vs_actual.csv", index=False)
    print("Results saved to outputs/predicted_vs_actual.csv")

if __name__ == "__main__":
    predict_race_positions(
        input_csv="data/features_with_position.csv",
        model_path="models/position_model.pkl"
    )
