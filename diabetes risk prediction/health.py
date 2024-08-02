# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\user\OneDrive\Documents\Desktop\diabetes risk prediction\diabetes.csv")

# Display basic information
print("Dataset Head:")
print(df.head())
print("\nDataset Description:")
print(df.describe())

# Define features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f'\nAccuracy: {accuracy_score(y_test, y_pred)}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save the model
joblib.dump(model, 'diabetes_model.pkl')
print("\nModel saved as 'diabetes_model.pkl'")

# Load the model (for later use)
loaded_model = joblib.load('diabetes_model.pkl')
print("\nModel loaded successfully")
