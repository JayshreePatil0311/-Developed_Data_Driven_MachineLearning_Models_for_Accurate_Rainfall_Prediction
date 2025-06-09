# 🌧️ DevelopedData-Driven MachineLearning Models for Accurate Rainfall Predicion
This project focuses on predicting rainfall using various machine learning algorithms. The goal is to forecast whether it will rain the next day using weather data. The dataset is analyzed and preprocessed before applying ML models like Logistic Regression, Decision Tree, Random Forest, etc., to compare their performance.

# 🎯 Project Objective
To build a predictive model using machine learning that forecasts whether it will rain tomorrow, based on today’s weather data such as temperature, humidity, wind speed, pressure, etc.

## 📊 Dataset Description
Source: Kaggle / Meteorological Department Dataset (CSV format)
Rows: 14,000+ entries
Features:Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, Humidity9am, Humidity3pm, Pressure9am, Temp3pm, RainToday, RainTomorrow (Target)

## 💻 Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn (for EDA & visualizations)
-  Google Colab## 🔍 Model Building

## 🧹 Data Preprocessing

- Removed null/missing values
- Label encoding for categorical variables
- Converted `RainToday` and `RainTomorrow` into binary labels (Yes → 1, No → 0)
- Normalization of continuous variables
- Feature selection based on correlation
- 
## The following ML models were trained and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Naive Bayes
5. K-Nearest Neighbors (KNN)
6. Support Vector Machine (SVM)

## 📏 Evaluation Metrics
Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve & AUC Score
## ✅ Results
![image](https://github.com/user-attachments/assets/a232be7f-3cc6-4147-bd6a-e948605ec570)

## Insights
📈 The best performing model was XGBoost using all features:

✅ Accuracy: 92.65%

✅ F1 Score (True): 0.91

It outperformed all other models including ANN and Random Forest in both full feature and PCA-reduced modes.

📌 We also analyzed:

Model performance using selected features

Model behavior with PCA-transformed data

The trade-off between model complexity and accuracy

