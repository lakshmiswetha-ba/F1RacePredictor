import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_actual_vs_predicted(input_csv, output_path):
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["RacePosition", "PredictedPosition", "Driver"])

    df = df.sort_values("RacePosition")  # Sort by actual result

    drivers = df["Driver"]
    actual = df["RacePosition"]
    predicted = df["PredictedPosition"]

    x = range(len(drivers))  # For positioning bars

    plt.figure(figsize=(10, 8))
    plt.barh(x, actual, color="gray", alpha=0.7, label="Actual")
    plt.barh(x, predicted, color="dodgerblue", alpha=0.7, label="Predicted")

    plt.yticks(x, drivers)
    plt.xlabel("Position")
    plt.title("Actual vs Predicted Race Positions")
    plt.gca().invert_yaxis()
    plt.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_actual_vs_predicted(
        input_csv="outputs/predicted_vs_actual.csv",
        output_path="visuals/position_barchart.png"
    )
