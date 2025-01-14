import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris()

# Creating a DataFrame from the dataset
iris_df = pd.DataFrame(data.data, columns=data.feature_names)

# Adding the target column to the DataFrame
iris_df['label'] = data.target

# Dictionary to convert target integers to species names
species_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

# Map the target numbers to actual species names
iris_df['label'] = iris_df['label'].map(species_dict)

NUM_FEATURES = 4

# Display the first few rows of the DataFrame
print(iris_df.head())

# Preparing data for k-NN classification
features = iris_df.drop('label', axis=1)
labels = iris_df['label']

# Initialize lists to store k-values and their corresponding accuracy scores
neighbors_range = []
accuracy_list = []

# Evaluate k-NN classifier performance for different k values
for neighbors in range(1, 11):
    # Split the dataset into training and testing sets with random splitting
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
    
    # Initialize and train the k-NN classifier
    knn_model = KNeighborsClassifier(n_neighbors=neighbors)
    knn_model.fit(features_train, labels_train)
    
    # Evaluate the model's accuracy on the test data
    accuracy = knn_model.score(features_test, labels_test)
    print(f"k = {neighbors}: Accuracy = {accuracy}")
    neighbors_range.append(neighbors)
    accuracy_list.append(accuracy)

# Plot the accuracy as a function of the number of neighbors
plt.plot(neighbors_range, accuracy_list, marker='o')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("k-NN Classification: Accuracy vs. Number of Neighbors")
plt.grid(True)
plt.show()
