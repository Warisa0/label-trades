# label-trades
# Trade Classification Using RandomForestClassifier

## Overview
This project implements a machine learning model to classify cryptocurrency trades as "Good" or "Bad" based on historical trade data. The model is built using Python with the **RandomForestClassifier** from Scikit-Learn. The dataset contains information on trade prices, volatility, and volume, which are used as features to train the model.

## Dataset
- **Source:** `crypto_trades.csv`
- **Columns:**
  - `Date`: Date of the trade.
  - `Open`: Opening price.
  - `High`: Highest price during the day.
  - `Low`: Lowest price during the day.
  - `Close`: Closing price.
  - `Volume`: Trading volume.
  - `Trade Outcome`: Classification label ("Good" or "Bad").

## Approach
1. **Data Preprocessing**
   - Convert the `Date` column to datetime format.
   - Sort data by date.
   - Feature engineering:
     - Calculate `Price Change %`: `(Close - Open) / Open * 100`
     - Calculate `High-Low %`: `(High - Low) / Low * 100`
     - Calculate `Volatility`: Rolling standard deviation of closing prices over 10 days.
   - Remove missing values.

2. **Feature Selection**
   - Features used for training: `Price Change %`, `High-Low %`, `Volume`, `Volatility`.
   - Target variable: `Trade Outcome` (converted to binary values: 1 for "Good", 0 for "Bad").

3. **Model Training & Evaluation**
   - Split data into training (80%) and testing (20%) sets.
   - Train a **RandomForestClassifier** with 100 estimators.
   - Predict on the test set and evaluate using:
     - **Accuracy Score**
     - **Classification Report**

4. **Feature Importance Analysis**
   - Plot feature importance scores to understand the impact of different factors in trade classification.

## Tools & Technologies
- **Programming Language:** Python
- **Libraries:**
  - `pandas`: Data manipulation
  - `numpy`: Numerical computations
  - `matplotlib` & `seaborn`: Data visualization
  - `sklearn`: Machine learning model and evaluation metrics

## Installation
Ensure all required libraries are installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Running the Code
1. Place `crypto_trades.csv` in the project directory.
2. Run the script using:
```bash
python project.py
```

## Results
- Model accuracy score displayed in the terminal.
- Classification report showing precision, recall, and F1-score.
- A bar plot visualizing feature importance in trade classification.

## Future Improvements
- Experiment with other machine learning models such as **XGBoost** or **Neural Networks**.
- Incorporate additional technical indicators like **Relative Strength Index (RSI)** and **Moving Averages**.
- Optimize hyperparameters for better model performance.

