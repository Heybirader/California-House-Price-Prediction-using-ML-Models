# California House Price Prediction using ML Models

Predicting the median house value of California census districts from geographical, demographic, and housing features, and evaluating four machine learning regression models: **Linear Regression**, **Random Forest**, **AdaBoost**, and **Gradient Boosting**.

---

## Dataset

- **Source:** California Housing dataset (1990 US Census), loaded from [`akmand/datasets`](https://github.com/akmand/datasets). Each row represents one census block group.
- **Raw Dimensions:** 20,640 rows × 10 columns (9 numerical features + 1 categorical feature).
- **Target Variable:** `median_house_value` (standardized during preprocessing).
- **Features:**
  - `longitude`, `latitude`: Geographic coordinates of the block group.
  - `housing_median_age`: Median age of houses within the block group.
  - `total_rooms`: Total number of rooms in the block group.
  - `total_bedrooms`: Total number of bedrooms (contains missing values in raw data).
  - `population`: Total residents in the block group.
  - `households`: Total number of households.
  - `median_income`: Median income of households (in tens of thousands of USD).
  - `ocean_proximity`: Categorical location relative to the coast (`<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN`).
- **Local Copies:**
  - Raw data: [`data/california_housing.csv`](data/california_housing.csv)
  - Preprocessed data: [`data/processed.csv`](data/processed.csv)

---

## Project Structure

```text
regression1/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── california_housing.csv     # Raw dataset (20,640 rows)
│   └── processed.csv              # Cleaned, scaled, and encoded dataset (17,609 rows)
├── notebooks/
│   ├── 01_eda.ipynb               # Data inspection, imputation, IQR outlier cleaning, scaling & encoding
│   └── 02_modeling.ipynb          # Model training, train/test evaluation, metric visualization & GridSearchCV
└── reports/
    ├── model_metrics.png          # Model comparison bar charts across 6 metrics
    └── picture/
        └── model_metrics.png      # Original generated metric plot
```

---

## Workflow & Methodology

### 1. Exploratory Data Analysis & Preprocessing ([`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb))
- **Data Inspection:** Verified data types, non-null counts, and distributions via `.info()`, `.describe(include="all")`, `.nunique()`, and `.hist(bins=60)`.
- **Missing Value Imputation:** Found 207 missing values in `total_bedrooms` (out of 20,640 entries) and imputed them using the column median.
- **Feature Relationships:**
  - Plotted `median_income` vs. `median_house_value` colored by `ocean_proximity`, showing a strong positive correlation between household income and housing prices.
  - Plotted geographic scatter plot (`latitude` vs. `longitude` colored by `median_house_value`) revealing coastal clustering of high-value homes.
  - Plotted correlation heatmap (`sns.heatmap`) and full pairplots.
- **Outlier Detection & Removal (IQR Method):**
  - Identified and removed outliers across all numerical columns using a threshold of $1.5 \times \text{IQR}$.
  - Rows were reduced from **20,640** to **17,609** (3,031 outlier records removed, ~14.7%).
- **Feature Scaling:** Applied `StandardScaler` to all numerical features (including the target `median_house_value`), transforming them into zero-mean, unit-variance $z$-scores.
- **Categorical Encoding:** One-hot encoded `ocean_proximity` using `pd.get_dummies()`, producing 5 binary dummy indicators.
- **Export:** Saved the final 17,609 × 14 dataset to `data/processed.csv`.

---

### 2. Model Training & Evaluation ([`notebooks/02_modeling.ipynb`](notebooks/02_modeling.ipynb))
- **Train/Test Split:** 90/10 split using `train_test_split(x, y, test_size=0.1, random_state=42)`:
  - **Training set:** 15,848 samples, 13 features
  - **Test set:** 1,761 samples, 13 features
- **Models Evaluated:**
  1. `LinearRegression()`
  2. `RandomForestRegressor()`
  3. `AdaBoostRegressor()`
  4. `GradientBoostingRegressor()`
- **Evaluation Function:** Computed Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$) across both train and test splits:
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}, \quad \text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|, \quad R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
- **Hyperparameter Tuning:** Tuned `RandomForestRegressor` via `GridSearchCV` with 5-fold cross-validation over:
  - `n_estimators`: `[100, 200]`
  - `max_depth`: `[None, 10, 20]`

---

## Results

*Note: Since the target `median_house_value` was standardized using `StandardScaler` in preprocessing, RMSE and MAE values are expressed in standardized units ($\sigma$), where lower values indicate better precision.*

| Model | Train RMSE | Train MAE | Train $R^2$ | Test RMSE | Test MAE | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (LR)** | 0.6128 | 0.4574 | 0.6234 | 0.6285 | 0.4573 | 0.6154 |
| **Random Forest (RF)** | **0.1710** | **0.1135** | **0.9707** | **0.4676** | **0.3061** | **0.7871** |
| **AdaBoost** | 0.8385 | 0.7328 | 0.2948 | 0.8412 | 0.7279 | 0.3111 |
| **Gradient Boosting (GradBoost)** | 0.5032 | 0.3590 | 0.7460 | 0.5366 | 0.3760 | 0.7196 |

### Hyperparameter Tuning (Random Forest)
- **Best Parameters:** `{'max_depth': None, 'n_estimators': 200}`
- **Best 5-Fold CV $R^2$ Score:** `0.7847` (0.78466)

---

## Model Comparison & Key Takeaways

![Model Metrics](reports/model_metrics.png)

1. **Top Performer — Random Forest:**
   - Achieved the highest test $R^2$ score (**0.7871**) and lowest test error (Test RMSE: **0.4676**, Test MAE: **0.3061**).
   - Showed high training fit ($R^2 = 0.9707$), indicating strong capacity to capture non-linear interactions across geographical coordinates and demographic indicators.
2. **Second Best — Gradient Boosting:**
   - Solid performance with a Test $R^2$ of **0.7196** and Test RMSE of **0.5366**, demonstrating balanced bias-variance trade-off without extreme training overfit.
3. **Linear Regression Baseline:**
   - Provided a reliable benchmark with Test $R^2 = 0.6154$ and consistent train/test performance (Train RMSE 0.6128 vs. Test RMSE 0.6285).
4. **Underperformer — AdaBoost:**
   - Performed poorly on this dataset with Test $R^2 = 0.3111$ and high error (Test RMSE: 0.8412), struggling to handle complex multivariate interactions compared to tree bagging and gradient boosting.
5. **Feature Dynamics:**
   - Median income, coastal proximity (`INLAND` vs. ocean proximity), and spatial coordinates (`latitude`, `longitude`) served as the primary determinants of price variance.

---

## Limitations

- **Price Capping:** The raw dataset caps `median_house_value` at \$500,000, creating an artificial upper boundary that constrains high-end predictions.
- **Target Standardization:** Preprocessing scaled the target variable with `StandardScaler` alongside features, meaning metric outputs reflect standardized deviation units rather than raw dollar amounts.
- **Data Leakage in Preprocessing:** Imputation, IQR outlier filtering, and feature scaling were executed across the entire dataset prior to the 90/10 train/test split. In production, these transformers should be fitted exclusively on training splits inside a `Pipeline`.
- **Spatial Autocorrelation:** A random train/test split allows adjacent geographical blocks to appear in both sets, potentially inflating spatial predictive accuracy.
- **Historical Data:** The dataset originates from the 1990 US Census and does not reflect modern real estate valuations.

---

## Future Improvements

- Wrap preprocessing inside a scikit-learn `Pipeline` and `ColumnTransformer` to prevent data leakage between train and test sets.
- Invert target scaling before computing test metrics to report errors directly in USD (\$ MAE, \$ RMSE).
- Engineer domain features: rooms per household (`total_rooms / households`), bedrooms per room (`total_bedrooms / total_rooms`), and population per household.
- Benchmark modern gradient boosting implementations: **HistGradientBoostingRegressor**, **XGBoost**, and **LightGBM**.
- Apply spatial / block-based cross-validation to assess generalization to unseen geographic regions.

---

## How to Run

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Heybirader/California-House-Price-Prediction-using-ML-Models.git
cd California-House-Price-Prediction-using-ML-Models
pip install -r requirements.txt
```

### 2. Execute Notebooks
Run the notebooks in sequential order:
1. **`notebooks/01_eda.ipynb`**: Performs exploratory data analysis, outlier cleaning, scaling, and generates `data/processed.csv`.
2. **`notebooks/02_modeling.ipynb`**: Loads processed data, splits train/test sets, fits all 4 regressors, renders metric charts, and executes GridSearchCV.

---

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, scikit-learn, Matplotlib, Seaborn