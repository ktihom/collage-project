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
