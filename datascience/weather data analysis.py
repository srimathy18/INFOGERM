# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
from google.colab import files 
uploaded = files.upload()

# Load the dataset
weather_df = pd.read_csv(next(iter(uploaded)))  

# Display basic information about the dataset
print(weather_df.info())
print(weather_df.describe())

print(weather_df.isnull().sum())

weather_df.fillna(method='ffill', inplace=True)


if 'Date/Time' in weather_df.columns:
    weather_df['Date/Time'] = pd.to_datetime(weather_df['Date/Time'])
if 'Weather' in weather_df.columns:
    le = LabelEncoder()
    weather_df['Weather'] = le.fit_transform(weather_df['Weather'])

# Correlation matrix to understand the relationships between variables
plt.figure(figsize=(10, 6))
corr_matrix = weather_df.corr()  
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Weather Variables')
plt.show()

# Exploratory Data Analysis (EDA)

plt.figure(figsize=(10, 6))
sns.histplot(weather_df['Dew Point Temp_C'], bins=30, kde=True, color='blue')
plt.title('Dew Point Temperature Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(weather_df['Rel Hum_%'], bins=30, kde=True, color='green')
plt.title('Relative Humidity Distribution')
plt.show()

# Time series analysis of temperature over time (if date is present)
if 'Date/Time' in weather_df.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(weather_df['Date/Time'], weather_df['Dew Point Temp_C'], label='Dew Point Temp')
    plt.xlabel('Date')
    plt.ylabel('Dew Point Temp (°C)')
    plt.title('Dew Point Temperature Over Time')
    plt.show()


if 'Dew Point Temp_C' in weather_df.columns and 'Rel Hum_%' in weather_df.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(weather_df['Date/Time'], weather_df['Dew Point Temp_C'], label='Dew Point Temp', color='orange')
    plt.plot(weather_df['Date/Time'], weather_df['Rel Hum_%'], label='Relative Humidity', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('Dew Point Temp and Relative Humidity Over Time (Impact on Agriculture)')
    plt.legend()
    plt.show()

# Forecasting Dew Point Temperature using ARIMA model
if 'Date/Time' in weather_df.columns and 'Dew Point Temp_C' in weather_df.columns:
    # Set the date as index
    weather_df.set_index('Date/Time', inplace=True)

    # Train ARIMA model for temperature forecasting
    temp_series = weather_df['Dew Point Temp_C']
    model = ARIMA(temp_series, order=(5, 1, 0))  
    model_fit = model.fit()

    # Forecast the next 30 days
    forecast = model_fit.forecast(steps=30)
    forecast_dates = pd.date_range(start=temp_series.index[-1], periods=30, freq='D')

    # Plot the forecasted values
    plt.figure(figsize=(12, 6))
    plt.plot(temp_series.index, temp_series, label='Actual Dew Point Temp')
    plt.plot(forecast_dates, forecast, label='Forecasted Dew Point Temp', color='red')
    plt.xlabel('Date')
    plt.ylabel('Dew Point Temp (°C)')
    plt.title('Dew Point Temperature Forecast for the Next 30 Days')
    plt.legend()
    plt.show()

features = weather_df[['Dew Point Temp_C', 'Rel Hum_%', 'Visibility_km']] 
target = weather_df['Weather']  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)

# Feature importance
feature_importance = rf_model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=features.columns, y=feature_importance)
plt.title('Feature Importance in Predicting Weather Conditions')
plt.show()
