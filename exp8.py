from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Step 1: Load the data
data = load_iris()
X, y = data.data, data.target
y = [data.target_names[label] for label in y]  # Step 2: Set target names as strings

# Step 3: Train-test split (60-40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Step 4: Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Fit the DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = clf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred, labels=data.target_names)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)

# Save confusion matrix and accuracy to a file
with open("output_results.txt", "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix) + "\n")
    f.write(f"Accuracy: {accuracy}\n")

# Step 7: Print tree in text format
print("Decision Tree:\n")
print(export_text(clf, feature_names=data.feature_names))

# Save the tree in text format to a file
with open("decision_tree.txt", "w") as f:
    f.write(export_text(clf, feature_names=data.feature_names))

# Step 8: Export the tree figure
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.savefig("decision_tree.png")
plt.show()
