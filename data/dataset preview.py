import pandas as pd

# Load the dataset (make sure to use the correct file name)
df = pd.read_csv("/Users/Pirates/Desktop/ai-sound-deaction-system/collage-project/data/dataset.csv")

# Display the first few rows
print(df.head())

# Show basic dataset info
print(df.info())

# check class imbalance

import matplotlib.pyplot as plt

# Count the occurrences of each engine condition
class_counts = df["Engine_Condition"].value_counts()

# Plot the class distribution
plt.figure(figsize=(6, 4))
class_counts.plot(kind="bar", color=["blue", "orange", "red"])
plt.xlabel("Engine Condition")
plt.ylabel("Count")
plt.title("Distribution of Engine Conditions")
plt.show()

# Print the exact counts
print(class_counts)

# hande imbalance

from imblearn.over_sampling import SMOTE
from collections import Counter

# Separate features and target
X = df.drop(columns=["Engine_Condition"])
y = df["Engine_Condition"]

# Apply SMOTE
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check new class distribution
new_counts = Counter(y_resampled)
print("After SMOTE:", new_counts)


# test split

from sklearn.model_selection import train_test_split

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Check the shape
print("Train set size:", X_train.shape, y_train.shape)
print("Test set size:", X_test.shape, y_test.shape)

# Normalize the data

from sklearn.preprocessing import StandardScaler

# Initialize scaler
scaler = StandardScaler()

# Fit on training data and transform both train & test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check shape
print("Scaled Train Data Shape:", X_train_scaled.shape)
print("Scaled Test Data Shape:", X_test_scaled.shape)

# Train the module random forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)


# feature importance

import matplotlib.pyplot as plt
import numpy as np

# Get feature importances
importances = rf_model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[::-1]  # Sort in descending order

# Plot feature importances
plt.figure(figsize=(10, 5))
plt.title("Feature Importance - Random Forest")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.show()

# hyperperameter tuning 

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize model
rf = RandomForestClassifier(random_state=42)

# Grid search
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Train the model with best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# Predictions & Evaluation
y_pred_best = best_rf.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Improved Accuracy: {accuracy_best * 100:.2f}%")

# module catboost

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Initialize CatBoost model
cat_model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, verbose=False)

# Train the model
cat_model.fit(X_train[:5000], y_train[:5000])

# Predictions
y_pred = cat_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"CatBoost Accuracy: {accuracy * 100:.2f}%")
