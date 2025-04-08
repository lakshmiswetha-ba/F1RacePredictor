import pandas as pd

def merge_position_data(features_file, race_results_file, output_file):
    features = pd.read_csv(features_file)
    results = pd.read_csv(race_results_file)

    merged = pd.merge(features, results, on="DriverNumber", how="left")

    if merged["RacePosition"].isnull().any():
        print("Some drivers are missing RacePosition. Check if they did not finish or were not in the race.")

    merged.to_csv(output_file, index=False)
    print(f"Merged file saved: {output_file}")
    print(merged[["DriverNumber", "Driver", "Team", "RacePosition"]].head(10))

if __name__ == "__main__":
    merge_position_data(
        features_file="data/features_enhanced_final.csv",
        race_results_file="data/raw/monza_2023_race_positions.csv",
        output_file="data/features_with_position.csv"
    )
