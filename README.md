<div align="center">
  
# 🏡 Ames Housing Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

*A comprehensive Machine Learning project predicting residential property prices using the Ames Housing dataset.*

</div>

---

## 📖 Table of Contents
- [Objective](#-objective)
- [Dataset](#-dataset)
- [Machine Learning Task](#-machine-learning-task)
- [Models Implemented](#-models-implemented)
- [Evaluation Metrics](#-evaluation-metrics)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Results & Insights](#-results--insights)
- [Limitations](#-limitations)
- [Disclaimer](#-disclaimer)

---

## 🎯 Objective

The primary objective of this project is to build a robust **supervised Machine Learning regression system** capable of accurately predicting the expected sale price of a residential property based on a wide array of physical and locational characteristics.

A secondary, yet equally important objective, is to perform a **Valuation Feature Analysis**. This helps answer critical business questions by identifying which property characteristics (e.g., total living area, overall quality, neighborhood) are most influential in driving residential property prices in the real estate market.

## 📊 Dataset

This project utilizes the well-known **Ames Housing dataset**, which contains detailed real estate data from Ames, Iowa. It is widely recognized from the Kaggle competition *"House Prices - Advanced Regression Techniques"*.

- **Source / Download**: [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- **Target Variable**: `SalePrice` (The property's sale price in dollars)
- **Features**: 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa.

## 🤖 Machine Learning Task

**Supervised Regression** — The model learns from labeled historical sales data to predict a continuous numerical value (the sale price).

## 🧠 Models Implemented

To ensure the best predictive performance, multiple models were evaluated and compared:
1. **Dummy Regressor** *(Baseline to establish minimum performance)*
2. **Linear Regression** *(For linear relationships and interpretability)*
3. **Random Forest Regressor** *(An ensemble of decision trees to capture non-linear patterns)*
4. **Extra Trees Regressor** *(An extremely randomized tree ensemble to reduce variance)*
5. **Gradient Boosting Regressor** *(A powerful sequential ensemble method, which ultimately proved to be the best performing model after hyperparameter tuning)*

## 📈 Evaluation Metrics

The models were rigorously evaluated using standard regression metrics via $K$-Fold Cross-Validation:
- **MAE (Mean Absolute Error)**: Measures the average magnitude of the errors in a set of predictions, without considering their direction. It is highly interpretable (error in dollars).
- **RMSE (Root Mean Squared Error)**: A quadratic scoring rule that measures the average magnitude of the error. It gives a relatively high weight to large errors, making it useful when large prediction errors are undesirable.
- **R² (Coefficient of Determination)**: Explains the proportion of variance in the target variable that is predictable from the features (higher is better, max 1.0).

## 📁 Project Structure

```text
ames-housing-price-prediction/
│
├── data/
│   ├── train.csv                # Training dataset (includes SalePrice)
│   ├── test.csv                 # Test dataset
│   ├── sample_submission.csv
│   └── data_description.txt     # Detailed feature dictionary
│
├── documentation/
│   ├── Exploratory Data Analysis (EDA) Insight Log.pdf
│   ├── Preprocessing & Feature Engineering Log.pdf
│   └── Final Report.pdf
│
├── notebooks/
│   └── Ames_Housing_Price_Prediction.ipynb  # Core ML Pipeline
│
├── outputs/
│   └── figures/                 # Generated EDA and Evaluation plots
│       ├── target_distribution.png
│       ├── correlation_heatmap.png
│       ├── feature_importance.png
│       └── ...
│
├── README.md
├── requirements.txt
└── .gitignore
```

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/umandathathsarani/ames-housing-price-prediction.git
   cd ames-housing-price-prediction
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   - Download the dataset from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
   - Create a `data/` directory in the root of the project.
   - Place `train.csv`, `test.csv`, and `data_description.txt` inside the `data/` directory. *(Note: The `data/` folder is ignored by git to prevent committing large CSV files).*

4. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook notebooks/Ames_Housing_Price_Prediction.ipynb
   ```
   *Run all cells from top to bottom to reproduce the preprocessing, model training, and evaluation steps.*

## 💡 Results & Insights

After executing the complete machine learning pipeline and performing hyperparameter tuning, the final evaluation revealed the best performing model to be the **Tuned Gradient Boosting Regressor**. 

*Note: Actual numerical metrics (MAE, RMSE, R²) can be viewed directly within the executed Jupyter Notebook.*

### Key Business Findings (Feature Importance):
The valuation feature analysis highlighted that the most influential characteristics for predicting price are:
1. **Size**: Total living area (`TotalSF` and `GrLivArea`).
2. **Quality**: The overall material and finish quality of the house (`OverallQual`).
3. **Age**: The age of the property (`PropertyAge` / `YearBuilt`).

## ⚠️ Limitations

- **Market Context**: The dataset relies on historical property sales data. Machine learning models assume historical patterns will continue, making the model potentially sensitive to broad macroeconomic shifts (e.g., changing interest rates, inflation).
- **Geographic Scope**: The predictions and feature importances are geographically limited to Ames, Iowa. The model cannot be applied directly to other cities or regions without retraining on local data.
- **Prediction Uncertainty**: There is inherent uncertainty in the model's predictions, particularly for properties at the extreme low or high ends of the price spectrum (luxury homes or highly dilapidated properties).

## ⚖️ Disclaimer

The predictions generated by this machine learning model are estimates based entirely on historical data patterns. They should be used strictly as a supplementary tool to guide analysis and should **not** be treated as guaranteed property valuations or formal financial/real estate advice.
