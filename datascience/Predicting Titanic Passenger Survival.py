# Install necessary libraries
# pip install pandas numpy seaborn matplotlib scikit-learn kaggle

# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss, matthews_corrcoef, roc_curve

train_df = pd.read_csv('titanic.csv')
print("Initial Dataset Information:\n", train_df.info())
print("Dataset Description:\n", train_df.describe())
print("Missing Values:\n", train_df.isnull().sum())
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
train_df.drop(columns='Cabin', inplace=True)
print("Missing values after initial preprocessing:\n", train_df.isnull().sum())
train_df.dropna(inplace=True)

# Drop non-numeric columns that are irrelevant for correlation
numeric_df = train_df.select_dtypes(include=[np.number])

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Exploratory Data Analysis (EDA) - survival by gender and class
sns.countplot(x='Survived', hue='Sex', data=train_df)
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=train_df)
plt.show()

# Convert categorical columns
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)

# Features and target selection
X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = train_df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RandomForest model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Model evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Additional metrics
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])  # AUC for binary classification
logloss = log_loss(y_test, model.predict_proba(X_test_scaled))  # Log loss for classification
mcc = matthews_corrcoef(y_test, y_pred)  # Matthews Correlation Coefficient

# Print the basic metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Print the additional metrics
print(f"ROC-AUC Score: {roc_auc}")
print(f"Log Loss: {logloss}")
print(f"Matthews Correlation Coefficient (MCC): {mcc}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
