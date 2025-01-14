import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 1: Generate data points with make_classification
data, labels = make_classification(n_samples=1000, n_features=4, n_classes=3, n_informative=3, n_redundant=0, random_state=42)

# Step 2: Scatter plot of features 1 & 2, and features 3 & 4
plt.figure(figsize=(12, 6))

# Scatter plot of feature 1 and 2
plt.subplot(1, 2, 1)
for class_value in np.unique(labels):
    plt.scatter(data[labels == class_value, 0], data[labels == class_value, 1], label=f"Class {class_value}")
plt.title("Feature 1 vs Feature 2")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# Scatter plot of feature 3 and 4
plt.subplot(1, 2, 2)
for class_value in np.unique(labels):
    plt.scatter(data[labels == class_value, 2], data[labels == class_value, 3], label=f"Class {class_value}")
plt.title("Feature 3 vs Feature 4")
plt.xlabel("Feature 3")
plt.ylabel("Feature 4")
plt.legend()

plt.tight_layout()
plt.show()

# Step 3: Train-test split (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

# Step 4: Fit GaussianNB model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 5: Predict on test data and print confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=np.unique(labels))
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
cmd.plot(cmap="viridis")
plt.title("Confusion Matrix")
plt.show()

# Step 6: Predict on three random data points
random_data_points = X_test[np.random.choice(X_test.shape[0], 3, replace=False)]
random_predictions = model.predict(random_data_points)
print("Random Data Points Predictions:", random_predictions)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 1: Generate data points with make_classification
data, labels = make_classification(n_samples=1000, n_features=4, n_classes=3, n_informative=3, n_redundant=0, random_state=42)

# Step 2: Scatter plot of features 1 & 2, and features 3 & 4
plt.figure(figsize=(12, 6))

# Scatter plot of feature 1 and 2
plt.subplot(1, 2, 1)
for class_value in np.unique(labels):
    plt.scatter(data[labels == class_value, 0], data[labels == class_value, 1], label=f"Class {class_value}")
plt.title("Feature 1 vs Feature 2")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# Scatter plot of feature 3 and 4
plt.subplot(1, 2, 2)
for class_value in np.unique(labels):
    plt.scatter(data[labels == class_value, 2], data[labels == class_value, 3], label=f"Class {class_value}")
plt.title("Feature 3 vs Feature 4")
plt.xlabel("Feature 3")
plt.ylabel("Feature 4")
plt.legend()

plt.tight_layout()
plt.show()

# Step 3: Train-test split (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

# Step 4: Fit GaussianNB model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 5: Predict on test data and print confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=np.unique(labels))
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
cmd.plot(cmap="viridis")
plt.title("Confusion Matrix")
plt.show()

# Step 6: Predict on three random data points
random_data_points = X_test[np.random.choice(X_test.shape[0], 3, replace=False)]
random_predictions = model.predict(random_data_points)
print("Random Data Points Predictions:", random_predictions)
