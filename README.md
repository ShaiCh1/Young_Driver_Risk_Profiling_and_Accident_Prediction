# Predicting Road Accidents Among Young Drivers

## Overview
This project focuses on predicting whether a young driver is likely to be involved in a road accident, using demographic data, licensing history, prior violations, and parental driving records. The goal is to develop a complete machine learning pipeline that includes clustering, data preprocessing, feature engineering, modeling, and evaluation.

---

## Goals
- Create behavioral profiles of young drivers using unsupervised learning techniques.
- Train a classifier to predict accident involvement.
- Handle real-world, multi-source data with varying quality and structure.
- Deal with class imbalance, feature leakage, and categorical encoding effectively.

---

## Dataset Description

Three data sources were integrated:

- **Drivers Information (K1)**: Demographics, licensing dates, restrictions, and place of residence.
- **Accidents (K2)**: Involvement in traffic accidents, type and severity.
- **Traffic Violations (K3)**: Type and timing of driving offenses.

All data is anonymized with consistent ID matching between the sources.

---

## Key Steps

### 1. Data Cleaning and Integration
- Merged all data sources using unique driver IDs.
- Filtered invalid records and handled missing values through logical imputation.
- Dropped features with more than 75% missing values.
- Removed data leakage features (e.g., accident severity or vetek post-accident).

### 2. Feature Engineering
- Constructed features like:
  - Driving experience (`overall_experience`)
  - Total number of violations
  - Parental violation/accident history
  - Categorical binning (e.g., continent of birth, district)
- Grouped rare categories and applied appropriate imputation strategies.

### 3. Clustering Young Drivers
- Built 4 clusters of driver profiles based on behavioral and demographic features.
- Characterized each cluster based on experience, risk indicators, and restrictions.
- Used the clusters as part of the downstream predictive analysis.

### 4. Classification Model
- Target variable: `has_accident` (binary).
- Addressed strong class imbalance using **SMOTE**.
- Applied a **Random Forest Classifier**.
- Evaluated performance on separate validation and test sets using:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion matrix

---

## Results

- The model achieved strong performance across all metrics, even with imbalanced classes.
- Clustering revealed clear risk segments among drivers.
- Feature engineering (e.g., encoding, binning) significantly improved interpretability and performance.


