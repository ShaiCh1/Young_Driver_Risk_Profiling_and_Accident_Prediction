# Young_Driver_Risk_Profiling_and_Accident_Prediction
---

## Young Driver Risk Profiling and Accident Prediction

This project demonstrates a full machine learning workflow aimed at analyzing risk factors for young drivers and predicting accident involvement. The solution integrates multiple data sources related to demographics, driving behavior, and traffic violations.

### Project Overview

The analysis includes:
- Data integration from three separate datasets:
  - **Drivers dataset** – contains demographic information, license history, and restrictions.
  - **Accidents dataset** – includes records of accidents and circumstances.
  - **Violations dataset** – contains traffic violations by the driver or their parents.

### Objectives

1. **Driver Profiling**  
   Use clustering techniques to group young drivers into distinct behavior/risk profiles based on a variety of features including experience, restrictions, and parental history.

2. **Accident Risk Prediction**  
   Train a supervised model to predict whether a young driver is likely to be involved in an accident.

### Key Steps

- Data cleaning and preparation
- Feature engineering, including handling of missing values and encoding of categorical variables
- Clustering to identify driver profiles
- Data balancing using SMOTE
- Model training using Random Forest
- Evaluation using classification metrics and confusion matrix

### Technologies Used

- Python
- pandas, NumPy
- scikit-learn
- imbalanced-learn
- matplotlib, seaborn

---
