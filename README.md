# 🏎️ Predicting Formula 1 Podium Finishers using Machine Learning


## Project Overview

The primary objective of this project is to develop a machine learning system that predicts the **Top 3 finishers (podium positions)** in Formula 1 Grand Prix races. By analyzing a combination of **historical and real-time data** — including driver statistics, lap times, tyre strategy, qualifying positions, and team performance — the model aims to deliver accurate race forecasts that can benefit **fans, analysts, and fantasy league participants**.

Through comprehensive data engineering and machine learning techniques, this project highlights the key factors that contribute to race outcomes and offers an interactive prediction tool.

---

## Data Collection & Exploratory Data Analysis

Collected and processed data using the **FastF1 API**, which provides real-time telemetry, lap-by-lap data, race metadata, and driver information. The following datasets were used:

- Race metadata (driver, constructor, grid position, track)
- Qualifying and lap time data
- Tyre compounds and stint information
- Pit stop timings
- Weather and circuit conditions (where available)

**Exploratory Data Analysis (EDA)** was performed to:
- Understand how starting position affects podium finishes
- Analyze tyre strategy vs. lap time consistency
- Compare driver and constructor performance across circuits
- Identify features with high predictive power

This initial analysis guided the feature engineering and model selection processes.

---

## Machine Learning Models

Trained and evaluated multiple classification models to identify the most accurate predictors for podium finishes. Models explored include:

- **Logistic Regression**
- **Random Forest**
- **XGBoost Classifier**

These models were selected based on their classification strength, interpretability, and proven performance in structured data problems.

---

## Application of Machine Learning Models

Each model was trained to predict whether a driver would **finish in the Top 3**. I engineered key features such as:

- Grid position, qualifying performance
- Average lap time in practice sessions
- Pit stop efficiency and tyre strategy
- Historical driver-track performance

Applied **k-fold cross-validation** and **grid search** to tune hyperparameters and evaluate model robustness.

**Evaluation Metrics:**
- Accuracy
- F1 Score
- Confusion Matrix


---

## Tech Stack

- **Python 3.9+**
- **FastF1 API** – F1 race data source
- **Pandas, NumPy** – Data manipulation
- **Scikit-learn, XGBoost** – Machine learning models
- **Matplotlib ** – Data visualization


---

##  Key Features

- Predicts Top 3 podium finishers using historical + live features
- Clean and modular codebase for training, prediction, and visualization
- Visual reports comparing **Predicted vs. Actual** results
- Feature importance analysis to interpret model decisions
- Extendable to predict full grid or DNF probabilities

---

##  Future Enhancements

- Include **live weather conditions** and session-based form
- Streamlit app to enable real-time user inputs for predictions
- Integrate race strategy simulations (e.g., tyre compound impact)
- Add support for predicting **Did Not Finish (DNF)** probabilities

---

## Developed By

- **Lakshmi Swetha Bathula**
  Data Scientist | ML Enthusiast | F1 Fan  
  [LinkedIn](https://www.linkedin.com/in/swethal-bathulaaa/) 

---

## Keywords

Formula One, F1 ML, podium prediction, motorsport analytics, driver performance, constructor performance, machine learning, classification, race prediction, FastF1, telemetry, tyre strategy, grid position, pit stop analysis, model interpretability, data visualization, EDA, feature engineering, sports data science.
