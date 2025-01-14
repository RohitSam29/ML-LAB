import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Download the IRIS Flower dataset and read it into a data frame
url = "iris_dataset.csv"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris_df = pd.read_csv(url, header=None, names=columns)

# 2. Pre-processing

# i. Transform the labels into integer form
label_encoder = LabelEncoder()
iris_df['species'] = label_encoder.fit_transform(iris_df['species'])

# ii. Normalize the data in each column
scaler = MinMaxScaler()
features = iris_df.iloc[:, :-1]
iris_df.iloc[:, :-1] = scaler.fit_transform(features)

# iii. Create an 80-20 train-test split of the dataset
X = iris_df.iloc[:, :-1].values
y = iris_df['species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create a MLP with one hidden layer with 10 nodes
model = Sequential([
    Dense(10, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(3, activation='softmax')  # 3 output classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Train the MLP and observe the loss
history = model.fit(X_train, y_train, epochs=100, verbose=0, validation_split=0.1)

# 5. Define a function for calculating accuracy
def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, np.argmax(y_pred, axis=1))

# 6. Print the confusion matrix
predictions = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, np.argmax(predictions, axis=1))
print("Confusion Matrix:")
print(conf_matrix)

# 7. Print the final accuracy
final_accuracy = calculate_accuracy(y_test, predictions)
print(f"Final Accuracy: {final_accuracy:.2f}")

# 8. Print the final connection weights
print("Final Connection Weights:")
for layer in model.layers:
    weights, biases = layer.get_weights()
    print("Weights:", weights)
    print("Biases:", biases)
