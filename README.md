# Reliance Stock Price Prediction using Machine Learning
This project predicts Reliance Industries stock closing prices using multiple machine learning algorithms. The best-performing model is automatically selected based on evaluation metrics and deployed through a Streamlit web application for interactive visualization and future price forecasting.

# Objective:-
To build a machine learning model that can accurately predict stock closing prices using historical market data and provide an easy-to-use interface for analysis and forecasting.

# Dataset:-
Company: Reliance Industries
Duration: One year of daily stock data (2024)
Source: NSE India (nseindia.com)
Target Variable: Close Price

# Workflow
Data Acquisition – Collect stock data from NSE
Preprocessing – Clean column names, remove unnecessary fields, handle missing values
Feature Engineering – Create new predictive features
Model Training – Train multiple ML algorithms
Evaluation – Compare models using RMSE and R² Score
Best Model Selection – Automatically choose the best model
Deployment – Visualize predictions using Streamlit

# Feature Engineering
Additional features created from raw data:
Day, Month, Year (from Date)
HL_Perc = (High − Low) / Low × 100
OC_Perc = (Close − Open) / Open × 100

Input Features (X):
Open, High, Low, Volume, Day, Month, Year, HL_Perc, OC_Perc

Target (y):
Close Price

# Machine Learning Models Used
Linear Regression
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest
Gradient Boosting
AdaBoost

The best model is automatically selected based on highest R² score and saved for deployment.

# Evaluation Metrics
R² Score – Measures prediction accuracy
RMSE (Root Mean Squared Error) – Measures prediction error

# Streamlit Application Features
Upload stock dataset
View Actual vs Predicted price trends
See interactive visualizations
Predict future closing prices based on selected date range
Download predicted results as CSV

# Limitations
Model trained on only one year of data
Not a time-series forecasting model
Future values of Open/High/Low/Volume are assumed constant
External factors (news, market sentiment, global events) are not included
Market volatility may not be fully captured
Some models may risk overfitting
Streamlit app does not retrain the model

# Technologies Used
Python
Pandas, NumPy
Scikit-learn
Streamlit
Jupyter Notebook

# How to Run
If you want to run the project locally:

pip install pandas numpy scikit-learn streamlit joblib
streamlit run predict.py

Ensure the following files are present in the same folder:
model.pkl
scaler.pkl
feature_cols.pkl

# Author
Pavithra L
MBA – Business Analytics & Finance
GitHub: pavithralanalytics

# Conclusion
This project demonstrates how machine learning can be effectively applied to stock market data for trend prediction. The automatic model selection and Streamlit deployment make the solution both accurate and user-friendly, showcasing strong analytical and technical skills.
