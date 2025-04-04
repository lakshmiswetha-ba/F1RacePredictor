import pandas as pd

# Step 1: Load the existing data
df = pd.read_csv("data/features_enhanced_final.csv")

# Step 2: Sort by AvgLapTime (lower = better)
df_sorted = df.sort_values(by="AvgLapTime").reset_index(drop=True)

# Step 3: Assign position (1 = fastest lap time)
df_sorted["RacePosition"] = df_sorted.index + 1

# Step 4: Save as a new CSV
df_sorted.to_csv("data/position_data.csv", index=False)

print(" RacePosition column added and saved as 'position_data.csv'")
