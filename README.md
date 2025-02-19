
![image](https://github.com/user-attachments/assets/91441e8a-9eda-4e37-9a9d-0f42ef450ed0)

# Milling Machine Failure Prediction
## Project Overview
This project focuses on predicting machine failures in a milling environment based on sensor data. The dataset includes various parameters such as rotational speed, torque, power, and tool wear, recorded over time until the failure occurs. The goal is to develop a predictive model that can accurately predict both the occurrence and type of machine failure, helping to prevent unexpected downtime and improve maintenance strategies.

## Table of Contents
- Introduction
- Dataset
- Data Preprocessing
- Modeling
- Evaluation
- Conclusion

## Introduction
The project aims to predict two key aspects:

- Machine Failure: Whether the machine will fail at any given time.
- Failure Type: Classifying the type of failure (e.g., TWF, HDF, PWF, OSF).
The data consists of several time series per machine, with each series having different time durations before the failure occurs.

# Dataset
Features: The dataset contains various machine parameters:

- Type of tool: L, M, H according his hardness
- Rotational speed: The speed of the machine's rotation.
- Torque: The amount of rotational force applied.
- Power: The energy consumed by the machine, calculated from rotational speed and torque.
- Tool wear: The wear level of the tool, which can indicate impending failure.

Target Variables:

- Machine Failure: Binary classification (0 = no failure, 1 = failure).
- Failure Type: Multi-class classification (TWF, HDF, PWF, OSF).

Imbalance: The dataset is imbalanced, with more instances of operational states than failures, which requires handling techniques like oversampling or undersampling.

## Data Preprocessing
- Handling Missing Values: Checked for NaNs and ensured data consistency.
- Feature Engineering: Applied techniques like feature scaling, encoding categorical variables, and handling imbalanced data.
- Resampling: Used oversampling (SMOTE) and undersampling strategies to balance the dataset for training models.
- PCA: Principal Component Analysis (PCA) was tested for dimensionality reduction to improve model performance, but did not yield better results than feature selection techniques like SelectKBest.

## Modeling
We used several machine learning models for classification, focusing on:

- SVC
- ANN
The models were evaluated based on metrics such as Accuracy, Recall, F1-Score, and ROC AUC.

## Model Selection Process:
- Grid Search: Fine-tuned hyperparameters using GridSearchCV to optimize the models.
- Cross-Validation: Used to assess the models' performance.

Recall-Focused: Given the importance of predicting failures (1), recall was the primary focus in the evaluation process.
Evaluation

## Conclusion
The chosen model provides a good balance between precision and recall, making it suitable for predicting machine failures in the milling environment. Although the model tends to predict more failures than actual (high recall), it is useful in preventing unexpected downtimes. However, there is a trade-off between prediction accuracy and production time, so careful consideration is required before deployment.

On the other hand, the use of neural networks gives very good results when classifying the types of failure.
