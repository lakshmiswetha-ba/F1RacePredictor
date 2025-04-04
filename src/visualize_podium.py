import pandas as pd
import matplotlib.pyplot as plt

# Load predicted and actual podium results
pred = pd.read_csv("outputs/predictions.csv")
actual = pd.read_csv("data/actual_results.csv")

# Merge both datasets on race name
merged = pd.merge(pred, actual, on="race")

# Create side-by-side bar plot
fig, ax = plt.subplots(figsize=(10, 4))
x = range(len(merged))

ax.bar([i - 0.2 for i in x], [1] * len(x), width=0.4, label="Actual", color="skyblue")
ax.bar([i + 0.2 for i in x], [1] * len(x), width=0.4, label="Predicted", color="salmon")

# Add podium labels above bars
for i, row in merged.iterrows():
    ax.text(i - 0.2, 1.02, row["podium_actual"], ha="center", fontsize=8, rotation=90)
    ax.text(i + 0.2, 1.02, row["podium_predicted"], ha="center", fontsize=8, rotation=90)

# Final touches
ax.set_xticks(x)
ax.set_xticklabels(merged["race"], rotation=45, ha='right')
ax.set_title("Predicted vs Actual Podium")
ax.set_ylim(0, 1.5)
ax.legend()

plt.tight_layout()
plt.savefig("visuals/predicted_vs_actual_podium.png")
plt.close()
