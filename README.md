# 🏎️ Predicting Formula 1 Podium Finishers using Machine Learning


Project Motivation

I’ve always been fascinated by how much strategy and data go into a Formula 1 race. It’s not just about speed — it’s about timing, precision, and decisions made in milliseconds. As a F1 🏎️ fan, I wanted to see if machine learning could make sense of all that complexity.

This project is a way to explore how data can tell the story of a race — and maybe even predict the outcome before the lights go out.

Project Summary

This project develops a machine learning pipeline to predict the Top 3 finishers (podium positions) in Formula 1 races using real-time and historical data from the FastF1 API. It features data extraction, preprocessing, feature engineering, model training (Random Forest, XGBoost), and performance evaluation through visual and quantitative methods.

Problem Statement

Formula 1 outcomes are influenced by multiple variables: driver skill, team strength, qualifying results, tyre strategy, and track conditions. The question is:

Can we use machine learning to accurately predict the top 3 finishers in a race using structured race data?

Approach

Data Extraction via FastF1 API

Feature Engineering: qualifying position, tyre compound, constructor stats, lap time averages

Modeling: Logistic Regression, Random Forest, XGBoost

Evaluation: accuracy, podium hit rate, visual validation

Visualization: predicted vs actual podium, feature importance

Key Results

Random Forest achieved ~65% accuracy in predicting podium finishes

Most influential features:

Qualifying Position

Constructor Points

Tyre Strategy

Visual comparisons helped validate model strengths and failure cases

Feature Importance

To understand how the model makes predictions, we analyzed feature importance scores from the Random Forest model:

Qualifying Position had the highest impact — better grid starts significantly correlate with podium finishes.

Constructor Points reflect overall team reliability and performance.

Tyre Strategy influenced pace and consistency throughout the race.

These results align with domain knowledge and improve model interpretability.



Predicted vs Actual Podium



📃 This comparison illustrates how closely model predictions match real race results. It helps identify races where predictions were accurate and where they diverged.

How to Run

1. Clone the Repository

git clone https://github.com/lakshmiswetha-ba/F1RacePredictor.git
cd F1RacePredictor

2. Install Dependencies

pip install -r requirements.txt

3. Run EDA Notebook

jupyter notebook notebooks/eda.ipynb

Explore, clean, and prepare the data.

4. Train the Model

python src/train_model.py

This will load the data, train the model, and generate predictions.

5. Review Outputs

Predictions: outputs/predictions.csv

Visualizations: visuals/

🛠 Tech Stack

Languages & Libraries: Python, pandas, scikit-learn, matplotlib, XGBoost

Data Source: FastF1 API

Development: Jupyter Notebooks, Python scripts

Features

Predicts Top 3 drivers for F1 races

Visual insights: feature importance, predicted vs actual podiums

Modular and extensible pipeline for experimentation

Future Enhancements

Include more historical data to improve generalization

Deploy as a Streamlit or Flask app for live predictions

Integrate additional features: pit stop strategy, weather, driver telemetry

Developed By

Lakshmi Swetha BGitHub | LinkedIn

What I Learned

Built a complete ML pipeline using real-world sports data

Explored race strategy through the lens of feature importance

Practiced model evaluation and storytelling for technical projects

Strengthened skills in data interpretation, visualization, and documentation

This project is open to collaboration and feedback. Feel free to fork, suggest improvements, or connect!