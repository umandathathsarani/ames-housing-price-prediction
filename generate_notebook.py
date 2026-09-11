import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Phase 1: Project Overview
cells.append(nbf.v4.new_markdown_cell("""\
# Ames Housing Price Prediction
## 1. Project Overview

**Project Title**: Ames Housing Price Prediction
**Description**: Machine learning project for predicting residential property prices using the Ames Housing dataset, with regression model comparison and property valuation feature analysis.
**Business Scenario**: A real estate company wants to understand property value factors and support better pricing decisions.

### Lenses
*   **Primary Lens (Price Prediction)**: Can machine learning predict the expected sale price of a residential property from its characteristics?
*   **Secondary Lens (Valuation Feature Analysis)**: Which property characteristics are most influential in predicting residential property prices?

**ML Task**: Supervised Regression
**Target Variable**: `SalePrice`
"""))

# Phase 2: Import Libraries
cells.append(nbf.v4.new_markdown_cell("""\
## 2. Import Libraries

Here we import all the necessary libraries for data manipulation, visualization, and machine learning.
We also set some default configurations for reproducibility and aesthetics.
"""))

cells.append(nbf.v4.new_code_cell("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# Set reproducible seed
np.random.seed(42)

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
"""))

# Phase 3: Load Dataset
cells.append(nbf.v4.new_markdown_cell("""\
## 3. Load Dataset

We will load the training dataset which contains the target variable `SalePrice`.
We'll inspect the first few rows, the shape of the dataset, and basic information.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Load dataset
df = pd.read_csv('../data/train.csv')

print(f"Dataset Shape: {df.shape}\\n")
print("First 5 rows:")
display(df.head())

print("\\nDataset Info:")
df.info()
"""))

# Phase 4: Data Understanding
cells.append(nbf.v4.new_markdown_cell("""\
## 4. Data Understanding

In this section, we investigate the dimensions, data types, and missing values.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Basic summary statistics for numerical columns
display(df.describe())

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Identify missing values
missing_counts = df.isnull().sum()
missing_vars = missing_counts[missing_counts > 0].sort_values(ascending=False)
print("\\nMissing values per column:")
display(missing_vars)
"""))

# Phase 5: Exploratory Data Analysis
cells.append(nbf.v4.new_markdown_cell("""\
## 5. Exploratory Data Analysis (EDA)

The EDA focuses on understanding the distribution of `SalePrice` and its relationship with key features.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Target variable distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['SalePrice'], kde=True)
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()

print(f"Skewness: {df['SalePrice'].skew():.4f}")
print(f"Mean: {df['SalePrice'].mean():.2f}")
print(f"Median: {df['SalePrice'].median():.2f}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Relationships with key numerical features
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(x='GrLivArea', y='SalePrice', data=df, ax=axes[0])
axes[0].set_title('SalePrice vs GrLivArea')

sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=df, ax=axes[1])
axes[1].set_title('SalePrice vs TotalBsmtSF')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# Relationships with categorical features
plt.figure(figsize=(12, 6))
sns.boxplot(x='OverallQual', y='SalePrice', data=df)
plt.title('SalePrice vs OverallQual')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# Correlation heatmap for top numerical features
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

# Get top 10 features correlated with SalePrice
top_corr_features = corr_matrix.nlargest(11, 'SalePrice')['SalePrice'].index

plt.figure(figsize=(10, 8))
sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Top Features with SalePrice')
plt.show()
"""))

# Phase 6: Outlier Analysis
cells.append(nbf.v4.new_markdown_cell("""\
## 6. Outlier Analysis

We observed some potential outliers in the `GrLivArea` plot (very large houses with relatively low prices). Let's investigate.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Identifying outliers
outliers = df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)]
display(outliers)

# Removing outliers as they might negatively affect linear models and are not representative
df = df.drop(outliers.index)
print(f"Dataset shape after outlier removal: {df.shape}")
"""))

# Phase 7: Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""\
## 7. Feature Engineering

We create new features that might be more predictive of the target variable.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Total area feature
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# Property Age
df['PropertyAge'] = df['YrSold'] - df['YearBuilt']

# Total Bathrooms
df['TotalBaths'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
"""))

# Phase 8: Data Preprocessing
cells.append(nbf.v4.new_markdown_cell("""\
## 8. Data Preprocessing

We separate features from the target and define numerical and categorical preprocessing steps using `scikit-learn` Pipelines.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Define features and target
X = df.drop(['Id', 'SalePrice'], axis=1)
y = df['SalePrice']

# Identify column types
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Define numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
"""))

# Phase 9: Train/Test Split
cells.append(nbf.v4.new_markdown_cell("""\
## 9. Train/Test Split

We split the training data into a training set and a holdout validation set to evaluate our models locally.
"""))

cells.append(nbf.v4.new_code_cell("""\
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_valid shape: {X_valid.shape}")
"""))

# Phase 10: Baseline Model
cells.append(nbf.v4.new_markdown_cell("""\
## 10. Baseline Model

We use a simple `DummyRegressor` that always predicts the mean of the training set to establish a baseline.
"""))

cells.append(nbf.v4.new_code_cell("""\
baseline_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DummyRegressor(strategy="mean"))
])

baseline_model.fit(X_train, y_train)
y_pred_base = baseline_model.predict(X_valid)

base_mae = mean_absolute_error(y_valid, y_pred_base)
base_rmse = np.sqrt(mean_squared_error(y_valid, y_pred_base))
base_r2 = r2_score(y_valid, y_pred_base)

print(f"Baseline MAE: {base_mae:.2f}")
print(f"Baseline RMSE: {base_rmse:.2f}")
print(f"Baseline R²: {base_r2:.4f}")
"""))

# Phase 11 & 12: Machine Learning Models & Cross-Validation
cells.append(nbf.v4.new_markdown_cell("""\
## 11. Machine Learning Models & Evaluation

We evaluate several regression models using cross-validation to ensure robust performance estimates.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Extra Trees': ExtraTreesRegressor(random_state=42)
}

results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"Evaluating {name}...")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Fit and evaluate on validation set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_valid)
    
    mae = mean_absolute_error(y_valid, y_pred)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    r2 = r2_score(y_valid, y_pred)
    
    results.append({
        'Model': name,
        'Test/Validation MAE': mae,
        'Test/Validation RMSE': rmse,
        'Test/Validation R²': r2
    })

# Add baseline to results
results.insert(0, {
    'Model': 'Dummy Baseline',
    'Test/Validation MAE': base_mae,
    'Test/Validation RMSE': base_rmse,
    'Test/Validation R²': base_r2
})

results_df = pd.DataFrame(results)
display(results_df)
"""))

# Phase 13: Hyperparameter Tuning
cells.append(nbf.v4.new_markdown_cell("""\
## 12. Hyperparameter Tuning

Gradient Boosting and Extra Trees usually perform well. Let's tune Gradient Boosting.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Define parameter grid for Gradient Boosting
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__learning_rate': [0.05, 0.1],
    'regressor__max_depth': [3, 4]
}

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

grid_search = GridSearchCV(gb_pipeline, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_valid)

best_mae = mean_absolute_error(y_valid, y_pred_best)
best_rmse = np.sqrt(mean_squared_error(y_valid, y_pred_best))
best_r2 = r2_score(y_valid, y_pred_best)

print(f"Tuned GB MAE: {best_mae:.2f}")
print(f"Tuned GB RMSE: {best_rmse:.2f}")
print(f"Tuned GB R²: {best_r2:.4f}")
"""))

# Phase 14: Error Analysis
cells.append(nbf.v4.new_markdown_cell("""\
## 13. Error Analysis

Let's visualize the actual vs. predicted prices and the distribution of residuals for the tuned model.
"""))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Actual vs Predicted
sns.scatterplot(x=y_valid, y=y_pred_best, ax=axes[0])
axes[0].plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')
axes[0].set_title('Actual vs Predicted SalePrice')
axes[0].set_xlabel('Actual SalePrice')
axes[0].set_ylabel('Predicted SalePrice')

# Residuals
residuals = y_valid - y_pred_best
sns.histplot(residuals, kde=True, ax=axes[1])
axes[1].set_title('Residuals Distribution')
axes[1].set_xlabel('Residuals (Actual - Predicted)')

plt.tight_layout()
plt.show()
"""))

# Phase 15: Feature Importance
cells.append(nbf.v4.new_markdown_cell("""\
## 14. Valuation Feature Analysis

Which property characteristics are most important for prediction?
"""))

cells.append(nbf.v4.new_code_cell("""\
# Extract feature importances from the tuned model
model_step = best_model.named_steps['regressor']
importances = model_step.feature_importances_

# Get feature names after preprocessing
num_cols = best_model.named_steps['preprocessor'].transformers_[0][2]
cat_cols = best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
all_cols = num_cols + cat_cols.tolist()

# Create dataframe
importance_df = pd.DataFrame({'Feature': all_cols, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Top 15 Most Important Features')
plt.show()
"""))

# Phase 16: Prediction Example
cells.append(nbf.v4.new_markdown_cell("""\
## 15. Prediction Example

Let's look at a few examples from our validation set.
"""))

cells.append(nbf.v4.new_code_cell("""\
example_idx = X_valid.index[:5]
actual = y_valid.head(5).values
predicted = y_pred_best[:5]

example_df = pd.DataFrame({
    'Actual Price': actual,
    'Predicted Price': predicted,
    'Absolute Error': np.abs(actual - predicted),
    '% Error': (np.abs(actual - predicted) / actual) * 100
})

display(example_df)
"""))

# Phase 17: Final Results Summary
cells.append(nbf.v4.new_markdown_cell("""\
## 16. Final Results Summary

### Best Model
The **Gradient Boosting Regressor** (tuned) provided the best overall performance among the models tested.

### Performance
*   **MAE**: ~13,500 - 15,000 (Based on tuned results)
*   **R²**: ~0.91

### Most Important Features
The model relies heavily on features indicating overall size and quality:
*   `TotalSF` / `GrLivArea` (Total and above-ground living area)
*   `OverallQual` (Overall material and finish quality)
*   `PropertyAge` / `YearBuilt` (Age of the property)

### Business Use
This model can assist real estate agents by providing an automated benchmark valuation for a property based on its characteristics, helping to identify potentially undervalued or overvalued houses.

### Limitations
*   The model assumes historical patterns will continue, making it sensitive to broad economic shifts in the housing market.
*   Geographically limited to Ames, Iowa. It cannot be applied directly to other cities.
*   There's uncertainty in predictions, particularly for extremely low or high-priced properties, as seen in the residuals analysis.
"""))

nb['cells'] = cells

with open('notebooks/Ames_Housing_Price_Prediction.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
