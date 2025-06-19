import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the CSV file
df = pd.read_csv('data/Asg6/wk6_assignment.csv')

# Extract the ground truth (true category) and predictions (predicted category)
y_true = df.iloc[:, 0]  # First column: true category
y_pred = df.iloc[:, 1]  # Second column: predicted category

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display the confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative (0)', 'Positive (1)'])

# Extract TN, FP, FN, TP from the confusion matrix
TN, FP, FN, TP = cm.ravel()  # Assumes the order is [0, 1] for labels

# Print the values
print("\nConfusion Matrix:")
print(f"True Negative (TN): {TN}")
print(f"False Positive (FP): {FP}")
print(f"False Negative (FN): {FN}")
print(f"True Positive (TP): {TP}")

# Plot the confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

