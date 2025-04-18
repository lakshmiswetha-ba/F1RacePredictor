# 🏎️ F1 Race Position Predictor

This project predicts the final race positions of Formula 1 drivers using real-world race data such as lap times, pit stops, and weather conditions. The model was trained using the 2023 Italian Grand Prix (Monza) data.

---

## Project Goal

To build a machine learning model that can predict each driver's finishing position in a race based on strategic and performance-related race features.

---

## Tools & Libraries

- Python
- pandas, numpy
- scikit-learn
- matplotlib
- joblib

---

## Features Used

- TotalLaps
- AvgLapTime
- NumPitStops
- AvgPitDuration
- SafetyCarLaps
- PitStopEfficiency
- LapTimeEfficiency
- WeatherCondition

---

## Model

- **Model Type**: Random Forest Regressor
- **Target Variable**: RacePosition (1 = 1st place)
- **Metric Used**: Mean Absolute Error (MAE)

**Final MAE on test data:** ~4.92 positions

## Visualizations

### Predicted vs Actual Race Positions

This bar chart compares predicted and actual race positions of drivers.

[View Race Position Comparison Chart](visuals/position_barchart.png)

---

### Feature Importance

This chart highlights the most important features used by the model for predicting race positions.

[View Feature Importance Chart](visuals/feature_importance.png)

---

## Folder Structure

F1-race-predictor/
├── data/
│   └── features_with_position.csv
├── models/
│   └── position_model.pkl
├── outputs/
│   └── predicted_vs_actual.csv
├── src/
│   ├── merge_driver_info.py
│   ├── merge_race_position.py
│   ├── plot_position_barchart.py
│   ├── predict_position.py
│   └── train_position_model.py 
├── visuals/
│   ├── feature_importance.png 
│   └── position_barchart.png
├── requirements.txt
└── README.md

---

## How to Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/lakshmiswetha-ba/F1RacePredictor.git
cd F1RacePredictor

---

**2. Install dependencies**  
Make sure you have Python installed, then install all required packages using:

```bash
pip install -r requirements.txt

---

**3. Run the model training and prediction scripts

python src/train_position_model.py
python src/predict_position.py
python src/plot_position_barchart.py

---

## What I Learned

Built a complete ML workflow using real-world race data

Practiced data cleaning, merging, and feature engineering

Trained and interpreted a regression model (Random Forest)

Learned to present results using clear visualizations

Gained experience writing technical documentation for a real-world project

---

## License

This project is licensed under the [MIT License](LICENSE).

---
