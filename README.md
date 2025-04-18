# 🏎️F1 Race Position Predictor

What happens when machine learning meets motorsport? This project explores that intersection—predicting the final race positions of Formula 1 drivers using real-world race data, and a sprinkle of predictive modeling magic.

##  Project Purpose

The core aim of this project was to build a machine learning pipeline capable of predicting the **finishing position** of each driver during a Grand Prix. Rather than rely on team standings or fan biases, the model analyzes **in-race performance data** to forecast final positions with surprising accuracy.

With F1 being a sport where strategy, milliseconds, and weather shifts make all the difference, this project tries to capture those nuances using data-driven techniques.

##  Data Source

All data was collected using the **[FastF1 API](https://pypi.org/project/fastf1/)**, a Python package that provides access to official F1 telemetry, timing, and race control data. Specifically, this model was trained using detailed telemetry from the **2023 Italian Grand Prix (Monza)**.

##  Tools & Libraries

- **Python**
- `pandas`, `numpy` – data manipulation
- `scikit-learn` – model training and evaluation
- `matplotlib` – visualizations
- `joblib` – model serialization

##  Features Used

The model leverages the following race-related features:

- `TotalLaps`
- `AvgLapTime`
- `NumPitStops`
- `AvgPitDuration`
- `SafetyCarLaps`
- `PitStopEfficiency`
- `LapTimeEfficiency`
- `WeatherCondition`

These were selected to reflect both race strategy and performance metrics.

##  The Model

- **Model Type**: Random Forest Regressor
- **Target Variable**: `RacePosition` (1 = 1st place)
- **Evaluation Metric**: Mean Absolute Error (MAE)

###  Result
Final MAE on the test set: **~4.92 positions**  
This means the model can generally predict a driver’s finishing position within ~5 spots of their actual result—a decent level of accuracy considering the unpredictable nature of F1 racing.

##  Visual Insights

Two key visualizations are included:

- **Predicted vs Actual Race Positions** – a bar chart comparing predictions with actual results
- **Feature Importance** – showing which features most influenced predictions

These charts can be found in the `visuals/` directory.

##  What I Learned

- Built a complete machine learning pipeline with real race telemetry
- Cleaned and engineered features from raw F1 race data
- Trained and evaluated a regression model using `scikit-learn`
- Visualized results to explain model decisions
- Wrote documentation to support reproducibility and understanding

##  Project Structure

F1-race-predictor/ ├── data/ # Processed datasets ├── models/ # Trained model artifacts ├── outputs/ # Model predictions ├── src/ # Source code (training, prediction, visualization) ├── visuals/ # Charts and visualizations ├── requirements.txt # Python dependencies └── README.md # Project overview and documentation


## License

This project is licensed under the **MIT License**. 



