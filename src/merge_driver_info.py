import pandas as pd

def merge_driver_info(features_file, race_file, output_file):
    """
    Merge driver names and team info into the features dataset based on DriverNumber.
    """

    # Load your engineered feature file
    features_df = pd.read_csv(features_file)

    # Load the driver/team mapping from FastF1 session data
    race_df = pd.read_csv(race_file)

    # Just keep the columns we care about
    race_df = race_df[["DriverNumber", "Driver", "Team"]]

    # Merge on DriverNumber
    merged_df = pd.merge(features_df, race_df, on="DriverNumber", how="left")

    # Save the result back to the features file (or to a new one)
    merged_df.to_csv(output_file, index=False)

    print(f"Merged driver info into: {output_file}")
    print(merged_df.head())

if __name__ == "__main__":
    merge_driver_info(
        features_file="data/features_enhanced_final.csv",
        race_file="data/raw/monza_2023_race_laptimes.csv",
        output_file="data/features_enhanced_final.csv"
    )
