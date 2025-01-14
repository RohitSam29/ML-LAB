from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Breast Cancer data
data = load_breast_cancer()

# 2. Display feature names, label names, and top 5 data points with labels
feature_names = data.feature_names
label_names = data.target_names
data_points = pd.DataFrame(data.data, columns=feature_names)
labels = data.target

print("Feature Names:")
print(feature_names)
print("\nLabel Names:")
print(label_names)
print("\nTop 5 Data Points with Labels:")
print(data_points.head(5))
print("\nLabels for Top 5 Data Points:")
print(labels[:5])

# 3. Train-test split (75-25)
X_train, X_test, y_train, y_test = train_test_split(data.data, labels, test_size=0.25, random_state=42)

# 4. Fit a linear SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# 5. Evaluate the performance on test data
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nPerformance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# 6. Visualization of Performance Metrics
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
scores = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))
sns.barplot(x=metrics, y=scores, palette="viridis")
plt.title("Performance Metrics of SVM Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.show()