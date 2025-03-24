# Young Drivers Risk Prediction

This project presents a complete end-to-end machine learning pipeline to analyze and predict accident risk among young drivers based on their demographic information, driving history, traffic violations, and familial background.

## Objective

The goal of this project is to:

1. **Segment young drivers** into meaningful risk groups using clustering techniques.
2. **Build predictive models** to assess the likelihood that a young driver will be involved in a traffic accident.

This solution simulates a real-world risk modeling pipeline using classical machine learning techniques with proper attention to data preprocessing, exploration, handling imbalanced classes, and model evaluation.

---

## Data Sources

The dataset includes three CSV files, each capturing a different aspect of a young driver's profile:

- **Drivers Information (`K1`)**: Demographic features, license issue dates, and parental identifiers.
- **Accidents (`K2`)**: Involvement in traffic accidents, vehicle types, severity, and context.
- **Violations (`K3`)**: Driving violations with metadata including timing and severity.

These datasets are linked by hashed driver IDs.

---

## Workflow Overview

### 1. Data Preparation & Cleaning

- Merged all data sources on unique driver identifiers.
- Cleaned column names, removed duplicates, and handled `NaN` values.
- Dropped columns with more than 75% missing data.
- Imputed missing values using logical defaults or grouped modes.

### 2. Feature Engineering

- Created features for:
  - Number and type of past accidents/violations
  - Parental driving behavior
  - Driving experience in days since license issuance
  - Demographic and geographic attributes
- Aggregated "settlement type", "continent", and "violation codes" into reduced meaningful categories.

### 3. Clustering (Task A)

- Used **KMeans** to segment drivers into four clusters based on demographic, behavioral, and driving features.
- Analyzed each group and labeled them with human-readable descriptions such as:
  - `Low Risk – No Incidents`
  - `High Risk – Multiple Violations`
  - `Moderate Risk – Low Experience`
  - `Potential Risk – Young with Risky Parents`

### 4. Predictive Modeling (Task B)

#### Target Variable

- The goal is to predict `has_accident` – a binary label indicating whether a driver has had an accident.

#### Preprocessing

- Applied **OneHotEncoding** to categorical features.
- Used `train_test_split` to divide the data into:
  - 70% training
  - 15% validation
  - 15% test

#### Class Imbalance

- The dataset was highly imbalanced (~85% of drivers had no accidents).
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to the training set to balance class distribution.

#### Models Used

- **Random Forest Classifier**
- **CatBoost Classifier** – chosen for its robust handling of categorical variables and imbalanced data.

Both models achieved high performance on validation and test sets, especially when trained on the SMOTE-balanced data.

### 5. Evaluation

- Performance metrics included **accuracy, precision, recall, and F1-score**.
- Also examined **confusion matrices** for a deeper view of prediction quality.
- Both models showed near-perfect results on test and validation sets due to strong signal from violation history, experience, and clustering.

---

## Highlights

- Robust handling of missing values and skewed distributions.
- Clear feature transformations and logical assumptions throughout.
- Applied both **unsupervised** and **supervised** ML techniques.
- Successfully handled imbalanced classification with SMOTE.
- Code is modular, reproducible, and well-commented.

---

## Future Improvements

- Try cost-sensitive learning instead of oversampling.
- Incorporate time series of violations/accidents over time.
- Add ensemble methods or stacking for improved generalization.

