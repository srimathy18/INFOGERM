# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from google.colab import files  

# Step 2: Upload the dataset
uploaded = files.upload()  
df = pd.read_csv(next(iter(uploaded)))  

# Display basic information about the dataset
print(df.info())
print(df.head())

# Step 3: Preprocess the text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    return text

# Apply preprocessing to the 'Text' column
df['Text'] = df['Text'].apply(preprocess_text)

# Step 4: Handle missing values in both text and label columns
df['Text'].replace('', np.nan, inplace=True)  
df.dropna(subset=['Text', 'label'], inplace=True) 

print(df.isnull().sum())

# Step 5: Encode the labels
df['label'] = df['label'].map({'Fake': 0, 'Real': 1})

# Step 6: Check for NaN values in 'label' column
print("Missing values in label column:", df['label'].isnull().sum())
df.dropna(subset=['label'], inplace=True)

# Step 7: Split the dataset into training and testing sets
X = df['Text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure there are no NaN values in the training and test sets
print("Missing values in X_train:", X_train.isnull().sum())
print("Missing values in y_train:", y_train.isnull().sum())
print("Missing values in X_test:", X_test.isnull().sum())
print("Missing values in y_test:", y_test.isnull().sum())

# Step 8: Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 9: Train various classification models and evaluate their performance

# Model 1: Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)
print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Model 2: Naive Bayes Classifier
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)
print("Naive Bayes Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

# Model 3: Support Vector Classifier (SVC)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)
print("Support Vector Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# Step 10: Visualize model performance
models = ['Random Forest', 'Naive Bayes', 'Support Vector']
accuracies = [
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_nb),
    accuracy_score(y_test, y_pred_svm),
]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies)
plt.title('Model Accuracies')
plt.ylabel('Accuracy')
plt.show()

# Step 11: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Random Forest Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', ax=axes[0])
axes[0].set_title('Random Forest Confusion Matrix')

# Naive Bayes Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', ax=axes[1])
axes[1].set_title('Naive Bayes Confusion Matrix')

# Support Vector Classifier Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', ax=axes[2])
axes[2].set_title('Support Vector Confusion Matrix')

plt.show()
