# Young Driver Risk Profiling and Accident Prediction

This project presents a full machine learning workflow for analyzing young drivers’ risk profiles and predicting their likelihood of being involved in a traffic accident. The data combines demographic information, driving history, accident records, and traffic violations to generate actionable insights and train predictive models.

## Project Structure

The project is divided into two main parts:

### 1. Driver Profiling with Clustering

- Goal: Identify distinct groups of young drivers based on their background and behavior.
- Methods: Clustering techniques (e.g., KMeans) were applied after extensive feature engineering to group drivers into meaningful profiles.
- Analysis: Each group was characterized by features such as driving experience, license restrictions, parental accident history, and demographic attributes.

### 2. Accident Risk Prediction

- Goal: Train a model to predict whether a young driver is likely to be involved in an accident.
- Model: Random Forest Classifier.
- Class Imbalance: Handled using SMOTE oversampling.
- Evaluation: Model performance evaluated on validation and test sets using accuracy, precision, recall, F1-score, and confusion matrix.

## Datasets

Three data sources were used:

- `K1`: Young drivers’ demographic and licensing data.
- `K2`: Accident records, including type and severity.
- `K3`: Traffic violation history for the driver and their parents.

These datasets were merged using a consistent anonymized driver ID.

## Key Techniques

- Data cleaning and imputation.
- Feature engineering and aggregation of accident/violation histories.
- Handling categorical variables via one-hot encoding.
- Train/Validation/Test splitting (70/15/15).
- Oversampling using SMOTE to address imbalanced classes.
- Clustering and group analysis using unsupervised learning.
- Model training and evaluation using scikit-learn tools.

## Technologies

- Python
- pandas, NumPy
- scikit-learn
- imbalanced-learn
- matplotlib, seaborn

## Results Summary

The final model achieved high performance on the test set, demonstrating strong predictive power, especially after addressing class imbalance and refining feature selection. Clustering added valuable interpretability by highlighting distinct behavioral patterns among different driver types.

## Note

All personal identifiers are anonymized. This project demonstrates best practices in building end-to-end ML pipelines with tabular data.

