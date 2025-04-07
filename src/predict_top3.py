import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Basic color map for F1 teams
TEAM_COLORS = {
    "Red Bull Racing": "blue",
    "Ferrari": "red",
    "Mercedes": "silver",
    "McLaren": "orange",
    "Aston Martin": "green",
    "Alpine": "purple",
    "AlphaTauri": "navy",
    "Alfa Romeo": "maroon",
    "Williams": "royalblue",
    "Haas F1 Team": "black"
}

def predict_and_plot_top3(input_csv, model_file, output_plot):
    # Load the feature data
    df = pd.read_csv(input_csv)

    # Keep driver and team info before dropping columns
    info_cols = df[["DriverNumber"]].copy()
    if "Driver" in df.columns and "Team" in df.columns:
        info_cols["Driver"] = df["Driver"]
        info_cols["Team"] = df["Team"]

    # Drop non-feature columns before prediction
    X = df.drop(columns=["AvgLapTime", "Driver", "Team"], errors="ignore")

    # Load the trained model
    model = joblib.load(model_file)

    # Predict lap times
    df["PredictedLapTime"] = model.predict(X)

    # Add driver and team info back
    if "Driver" in info_cols.columns:
        df = pd.concat([df, info_cols[["Driver", "Team"]]], axis=1)

    # Get top 3 drivers with lowest predicted lap times
    top3 = df.sort_values(by="PredictedLapTime").head(3).copy()

    # Print top 3 to console
    print("Top 3 Predicted Drivers:")
    print(top3[["Driver", "Team", "PredictedLapTime"]])

    # Build labels like "Max Verstappen (Red Bull Racing)"
    top3["Label"] = top3.apply(lambda row: f"{row['Driver']} ({row['Team']})", axis=1)

    # Get bar colors by team
    top3["Color"] = top3["Team"].map(TEAM_COLORS).fillna("gray")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(top3["Label"], top3["PredictedLapTime"], color=top3["Color"])
    plt.xlabel("Driver (Team)")
    plt.ylabel("Predicted Avg Lap Time")
    plt.title("Top 3 Predicted Fastest Drivers")
    plt.xticks(rotation=10)
    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot)
    print(f"Plot saved to: {output_plot}")
    plt.show()

if __name__ == "__main__":
    predict_and_plot_top3(
        input_csv="data/features_enhanced_final.csv",
        model_file="models/lap_time_model.pkl",
        output_plot="visuals/top3_predicted.png"
    )
