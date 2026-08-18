# Quality Prediction in Mining Project

> A time-series machine learning project that uses XGBoost regression to predict **% Silica Concentrate one hour ahead** from flotation plant process measurements.

## 📌 Project Overview

This project applies machine learning to a mining flotation process, with the goal of forecasting the future **% Silica Concentrate** value from process measurements.

The project:

* Downloads the mining dataset from Google Drive using `gdown`.
* Loads and cleans the flotation-process data.
* Converts the first column into a datetime index.
* Converts object/numeric columns into numeric values.
* Resamples the data to **1-minute intervals** using the mean.
* Handles missing values using forward fill followed by backward fill.
* Uses process variables as input features.
* Excludes `% Iron Concentrate` from the model inputs.
* Shifts the target by **60 minutes** to create a one-hour-ahead prediction task.
* Splits the time-series data into training and testing sets without shuffling.
* Trains an **XGBoost Regressor**.
* Evaluates the model using **RMSE** and **R²**.
* Generates feature-importance, actual-vs-predicted, time-series, and correlation visualizations.

The current notebook output reports an **RMSE of 1.2363** and an **R² of -0.0987** on the test set for the one-hour-ahead prediction task. These are the results saved in the uploaded notebook and should be interpreted as the current project's measured performance, not as a production benchmark.

---

## 🎯 Problem Statement

In a flotation-based mining process, the quality of the concentrate can change with operating conditions such as air flow, pulp properties, reagent flow, and flotation-column levels.

The project formulates this as a supervised time-series regression problem:

> **Given the current process measurements, predict the `% Silica Concentrate` approximately one hour into the future.**

Accurate forecasting could help provide earlier visibility into changes in concentrate quality and support future process-monitoring or optimization workflows.

---

## 💡 Objective

The primary objective is to build an XGBoost regression model that predicts the future **% Silica Concentrate** value using current flotation-process measurements.

The implemented forecasting horizon is:

**1 hour ahead = 60 minutes**

---

## ✨ Key Features

* Mining flotation-process data ingestion
* Google Drive dataset download using `gdown`
* Datetime parsing and indexing
* Numeric type conversion
* 1-minute resampling
* Missing-value handling with `ffill()` and `bfill()`
* Feature/target separation
* Exclusion of `% Iron Concentrate` from model inputs
* One-hour-ahead target shifting
* Time-order-preserving train/test split
* XGBoost regression
* RMSE and R² evaluation
* Top-10 feature-importance visualization
* Actual vs. predicted scatter plot
* 200-minute test-set time-series comparison
* Correlation heatmap for top features and the target

---

## 🧠 Machine Learning Approach

### Problem Type

**Supervised regression with a time-series forecasting setup.**

The model predicts a continuous target:

`% Silica Concentrate`

### Model

The implemented model is:

**XGBoost `XGBRegressor`**

Configuration used in the notebook:

```python
xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```

### Why XGBoost?

XGBoost is a gradient-boosted tree algorithm that can model nonlinear relationships between process variables and a continuous target. It is used here to learn relationships between the flotation-process measurements and the future silica-concentrate value.

### Training Strategy

The notebook:

1. Resamples the original data to one-minute intervals.
2. Creates a future target by shifting `% Silica Concentrate` by 60 rows.
3. Removes rows where the shifted target is unavailable.
4. Splits the resulting data using:

   * `test_size=0.2`
   * `shuffle=False`
5. Trains XGBoost on the earlier portion of the time-ordered data.
6. Evaluates predictions on the later test portion.

This avoids randomly shuffling the time-series observations during the train/test split.

---

## 📊 Dataset

### Dataset Name

The downloaded/extracted CSV used by the notebook is:

`MiningProcess_Flotation_Plant_Database.csv`

### Dataset Source

The notebook downloads the data from a Google Drive file using the following file ID:

`1N80d8eTDAf1JMQXGQbHDAUaMGRyA8QG3`

Download URL used by the notebook:

https://drive.google.com/uc?id=1N80d8eTDAf1JMQXGQbHDAUaMGRyA8QG3

### Dataset Format

CSV after extraction.

The notebook also contains logic to handle the downloaded file if it is actually a ZIP archive containing the CSV.

### Dataset Size

The notebook does not report the original raw-row count before resampling.

After 1-minute resampling and missing-value filling, the processed dataframe has:

* **264,841 rows**
* **23 columns**

### Target Variable

```text
% Silica Concentrate
```

### Excluded Column

The notebook explicitly excludes:

```text
% Iron Concentrate
```

The code comments describe this column as highly correlated and remove it before model training.

### Input Features

The model uses **21 input features**:

1. `% Iron Feed`
2. `% Silica Feed`
3. `Starch Flow`
4. `Amina Flow`
5. `Ore Pulp Flow`
6. `Ore Pulp pH`
7. `Ore Pulp Density`
8. `Flotation Column 01 Air Flow`
9. `Flotation Column 02 Air Flow`
10. `Flotation Column 03 Air Flow`
11. `Flotation Column 04 Air Flow`
12. `Flotation Column 05 Air Flow`
13. `Flotation Column 06 Air Flow`
14. `Flotation Column 07 Air Flow`
15. `Flotation Column 01 Level`
16. `Flotation Column 02 Level`
17. `Flotation Column 03 Level`
18. `Flotation Column 04 Level`
19. `Flotation Column 05 Level`
20. `Flotation Column 06 Level`
21. `Flotation Column 07 Level`

---

## 🔄 Data Preprocessing

The implemented pipeline is:

```text
Google Drive Dataset
        ↓
Download with gdown
        ↓
ZIP / CSV Handling
        ↓
CSV Loading
        ↓
Column Name Cleaning
        ↓
Datetime Conversion
        ↓
Drop Invalid Dates
        ↓
Set Date as Index
        ↓
Numeric Type Conversion
        ↓
1-Minute Mean Resampling
        ↓
Forward Fill + Backward Fill
        ↓
Separate Features and Target
        ↓
Drop % Iron Concentrate
        ↓
Shift Target by 60 Minutes
        ↓
Remove Invalid Shifted Targets
        ↓
Time-Ordered Train/Test Split
        ↓
XGBoost Regression
```

### 1. Column Cleaning

Column names are stripped of surrounding whitespace.

### 2. Timestamp Processing

The first dataframe column is treated as the date/time column.

It is converted using:

```python
pd.to_datetime(..., errors='coerce')
```

Rows with invalid timestamps are removed.

The timestamp column is then used as the dataframe index.

### 3. Numeric Conversion

Object columns have commas replaced with periods before conversion to floating-point values.

Other columns are converted using numeric coercion.

### 4. Resampling

The cleaned data is resampled to one-minute intervals:

```python
df.resample('1T').mean()
```

The mean value is calculated for each one-minute interval.

### 5. Missing-Value Handling

Missing values created or remaining after resampling are handled using:

```python
ffill().bfill()
```

That means forward filling is performed first, followed by backward filling.

### 6. Feature and Target Separation

The last column is used as the target:

```text
% Silica Concentrate
```

The second-last column:

```text
% Iron Concentrate
```

is removed from the feature matrix.

---

## ⏱️ Forecasting Setup

The project performs a **one-hour-ahead prediction**.

The notebook defines:

```python
hours_ahead = 1
steps = hours_ahead * 60
```

The future target is created using:

```python
y_shifted = y.shift(-steps)
```

Therefore:

```text
1 hour × 60 minutes = 60 rows
```

The model uses the current input features to predict the `% Silica Concentrate` value 60 minutes ahead.

---

## 🧪 Train/Test Split

The notebook uses:

```python
train_test_split(
    X_lagged,
    y_lagged,
    test_size=0.2,
    shuffle=False
)
```

The saved notebook output reports:

| Split    | Samples | Features |
| -------- | ------: | -------: |
| Training | 211,824 |       21 |
| Testing  |  52,957 |       21 |

Because `shuffle=False` is used, the chronological order is preserved during the split.

---

## 📈 Model Evaluation

The notebook evaluates the model using:

### Root Mean Squared Error (RMSE)

```text
RMSE = 1.2363
```

### R² Score

```text
R² = -0.0987
```

These values are the actual saved notebook results for the one-hour-ahead test set.

### Interpretation

The current measured R² is negative, which indicates that the trained model does not outperform the baseline represented by predicting the test-set mean under the R² definition.

The project should therefore be treated as an experimental/academic machine-learning implementation rather than a validated production forecasting system.

---

## 🔍 Feature Importance

The notebook calculates XGBoost feature importance and visualizes the top 10 features.

The saved project visualization ranks the following features highest:

| Rank | Feature                      |
| ---: | ---------------------------- |
|    1 | Flotation Column 01 Air Flow |
|    2 | Flotation Column 03 Air Flow |
|    3 | Flotation Column 04 Air Flow |
|    4 | Flotation Column 05 Level    |
|    5 | Amina Flow                   |
|    6 | Flotation Column 07 Level    |
|    7 | % Silica Feed                |
|    8 | Flotation Column 04 Level    |
|    9 | Ore Pulp pH                  |
|   10 | Flotation Column 03 Level    |

The chart displays the model's relative feature-importance scores.

Important: feature importance from this model indicates the features used most strongly by the trained XGBoost model; it does not by itself establish causal relationships.

---

## 🔗 Correlation Analysis

The notebook creates a correlation heatmap using the top 7 model-important features plus the target.

The saved visualization shows, among other relationships:

* `Flotation Column 01 Air Flow` and `Flotation Column 03 Air Flow`: **0.97**
* `Flotation Column 05 Level` and `Flotation Column 07 Level`: **0.83**
* `Flotation Column 01 Air Flow` and `Target_%_Silica`: **-0.33**
* `Flotation Column 03 Air Flow` and `Target_%_Silica`: **-0.33**
* `% Silica Feed` and `Target_%_Silica`: **0.03**

These are correlations calculated in the notebook's selected training-data visualization and should not be interpreted as causal effects.

---

## 📊 Visualizations

The project contains four saved visualization images:

### 1. Feature Importance

`m1.png`

Shows the top 10 features ranked by XGBoost relative importance.

### 2. Actual vs Predicted

`m2.png`

Shows the relationship between actual and predicted `% Silica Concentrate` values for the test set, together with a perfect-prediction reference line.

### 3. Time-Series Prediction Snapshot

`m3.png`

Compares actual and predicted `% Silica Concentrate` values for the first **200 minutes** of the test set.

### 4. Correlation Heatmap

`m4.png`

Shows correlations among the seven selected important features and the target.

---

## 📁 Project Structure

The uploaded project contains the following files:

```text
Quality-Prediction-In-minning-Project-main/
│
├── Quality_Prediction_In_minning_Project.ipynb
├── README.md
├── m1.png
├── m2.png
├── m3.png
└── m4.png
```

### File Description

| File                                          | Purpose                                                                    |
| --------------------------------------------- | -------------------------------------------------------------------------- |
| `Quality_Prediction_In_minning_Project.ipynb` | Complete data-processing, training, evaluation, and visualization workflow |
| `README.md`                                   | Project documentation                                                      |
| `m1.png`                                      | XGBoost top-10 feature-importance visualization                            |
| `m2.png`                                      | Actual vs predicted silica scatter plot                                    |
| `m3.png`                                      | 200-minute actual vs predicted time-series snapshot                        |
| `m4.png`                                      | Correlation heatmap                                                        |

The dataset CSV itself is **not included inside the uploaded project ZIP**. The notebook downloads it at runtime.

---

## 🛠️ Technologies & Libraries

The notebook installs and uses:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* gdown
* Google Colab

The installation cell contains:

```bash
pip install gdown xgboost scikit-learn pandas numpy matplotlib seaborn
```

---

## 🚀 How to Run

### Option 1 — Google Colab

The notebook contains a Google Colab launch link.

Open:

https://colab.research.google.com/github/purusottamdash0-collab/Quality-Prediction-In-minning-Project/blob/main/Quality_Prediction_In_minning_Project.ipynb

Then run the notebook cells from top to bottom.

### Option 2 — Local Jupyter Environment

Clone/download the repository containing the notebook and install the required packages:

```bash
pip install gdown xgboost scikit-learn pandas numpy matplotlib seaborn
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
Quality_Prediction_In_minning_Project.ipynb
```

Run the cells sequentially.

### Dataset Requirement

The notebook expects access to the Google Drive dataset referenced by its `gdown` file ID.

If the dataset is unavailable from that source, the notebook will not be able to reproduce the complete pipeline unless the expected `mining_data.csv`/CSV input is supplied manually in the working directory.

---

## 🔧 Implementation Notes

### Dataset Download

The notebook downloads the dataset to:

```text
mining_data.csv
```

It then checks whether the downloaded file is a ZIP archive.

If required, nested ZIP extraction is performed until a CSV is found.

### CSV Loading

The extracted CSV is loaded with:

```python
pd.read_csv(
    csv_path,
    encoding='latin1',
    sep=','
)
```

### Model Input

The model receives the 21 process features listed in the Dataset section.

No separate scaling/standardization step is implemented in the notebook.

No separate feature-engineering library or pipeline object is implemented.

---

## ⚠️ Current Limitations

Based strictly on the uploaded project:

* The current test-set **R² is -0.0987**, so the existing model is not yet a strong forecasting solution.
* No hyperparameter search or systematic model comparison is implemented.
* No cross-validation designed specifically for time-series data is implemented.
* No model persistence file such as `.pkl`, `.joblib`, or serialized XGBoost model is included.
* No standalone inference script is included.
* No REST API is included.
* No dashboard or web application is included.
* No Docker configuration is included.
* No `requirements.txt` file is included.
* The raw dataset is not included in the ZIP.
* The notebook uses the deprecated pandas resampling alias `'1T'`; newer pandas versions recommend `'1min'`.
* The project does not contain a separately implemented production preprocessing pipeline.
* The saved results are from the notebook execution and are not presented as independently reproduced benchmarks.

---

## 🚀 Potential Future Improvements

These are suggested improvements based on the current implementation; they are **not currently implemented features**.

### Modeling

* Perform systematic XGBoost hyperparameter tuning.
* Compare XGBoost with other regression baselines.
* Establish a simple baseline model before judging improvement.
* Explore lag features and rolling statistical features.
* Evaluate multiple forecasting horizons.
* Use time-series-aware cross-validation.

### Data & Features

* Perform more detailed missing-value and outlier analysis.
* Investigate feature redundancy and multicollinearity.
* Add domain-informed process features where appropriate.
* Validate whether the excluded `% Iron Concentrate` variable should remain excluded under a strict forecasting setup.

### Evaluation

* Add MAE and additional regression diagnostics.
* Evaluate performance across different time periods.
* Compare performance across multiple future horizons.
* Analyze prediction errors over time.
* Validate the model on a completely separate temporal holdout.

### Engineering

* Move preprocessing and inference into reusable Python modules.
* Add a `requirements.txt` file.
* Add automated tests.
* Add model serialization.
* Add a command-line inference workflow.
* Add experiment tracking.
* Add CI checks for the repository.

### Deployment

A future version could expose the trained model through an API or dashboard, but **no deployment/API/dashboard implementation exists in the current uploaded project**.

---

## 📚 Project Status

**Current status:** Experimental / academic machine-learning project.

The complete uploaded workflow demonstrates data ingestion, time-series preprocessing, one-hour target forecasting, XGBoost regression, evaluation, feature-importance analysis, and visualization.

The current saved evaluation:

```text
Forecast Horizon : 1 hour
Model            : XGBRegressor
RMSE             : 1.2363
R²               : -0.0987
```

Because the current R² is negative, further model development and validation are required before considering the system reliable for real-world production use.

---

## 👨‍💻 Author

**Purusottam Dash**

B.Tech Computer Science Engineering — Artificial Intelligence & Machine Learning

Focus areas include:

* Machine Learning
* Deep Learning
* Data Science
* Python
* AI/ML Engineering




If this repository is intended for public open-source use, a suitable license should be added separately.
::: 
