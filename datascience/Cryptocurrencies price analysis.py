import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import files

# Step 1: Upload the dataset
uploaded = files.upload()  
df = pd.read_csv(next(iter(uploaded)))

# Step 2: Data Preprocessing
# Check the structure of the dataset
print(df.head())
print(df.info())

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort the data by date
df = df.sort_values(by='Date')
X = df[['Open', 'High', 'Low', 'Volume', 'Marketcap']]  # Features
y = df['Close']  # Target variable

# Handle missing values (if any)
df.dropna(inplace=True)

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Building
# 1. Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
rf_pred = rf_reg.predict(X_test)

# 2. Support Vector Regressor (SVR)
svr_reg = SVR()
svr_reg.fit(X_train, y_train)
svr_pred = svr_reg.predict(X_test)

# Step 5: Model Evaluation
print("Random Forest Regressor Performance:")
print(f"Mean Squared Error: {mean_squared_error(y_test, rf_pred)}")
print(f"R-squared: {r2_score(y_test, rf_pred)}")

print("Support Vector Regressor Performance:")
print(f"Mean Squared Error: {mean_squared_error(y_test, svr_pred)}")
print(f"R-squared: {r2_score(y_test, svr_pred)}")

# Step 6: Visualization of Predictions
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Actual Close Price', color='blue')
plt.plot(rf_pred, label='Random Forest Predictions', color='green')
plt.plot(svr_pred, label='SVR Predictions', color='red')
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Index')
plt.ylabel('Close Price')
plt.legend()
plt.show()

#  Correlation Heatmap
numeric_df = df.select_dtypes(include=['float64', 'int64'])  

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Cryptocurrency Features')
plt.show()
