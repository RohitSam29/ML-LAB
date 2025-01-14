import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('sign_distance.csv')

print("First five rows of the dataset:")
print(df.head())

data = df.values

X = data[:, 0]
Y = data[:, 1]


N = len(X)

sum_x = np.sum(X)
sum_y = np.sum(Y)
sum_xy = np.sum(X * Y)
sum_x2 = np.sum(X ** 2)


m = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x ** 2)
b = (sum_y - m * sum_x) / N

# Step 3: Display the parameters of the regression line
print("\nManual Linear Regression Parameters:")
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

# Step 4: Plot the data points and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Data Points')

# Calculate the predicted Y values using the manual regression parameters
Y_pred_manual = m * X + b
plt.plot(X, Y_pred_manual, color='red', label='Manual Regression Line')

plt.title('Linear Regression: Manual Calculation')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (Y)')
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Predict the value of an unknown data point where X = 50
unknown_x = 50
predicted_y = m * unknown_x + b
print(f"\nPredicted Y value for X = {unknown_x}: {predicted_y}")

# Step 6: Perform linear regression using scikit-learn's LinearRegression and verify the results
# Reshape X for sklearn which expects a 2D array for features
X_reshaped = X.reshape(-1, 1)

# Create and fit the model
model = LinearRegression()
model.fit(X_reshaped, Y)

# Retrieve the slope and intercept from the model
m_sklearn = model.coef_[0]
b_sklearn = model.intercept_

print("\nScikit-learn Linear Regression Parameters:")
print(f"Slope (m): {m_sklearn}")
print(f"Intercept (b): {b_sklearn}")

# Verify that the manual and sklearn parameters are the same (or very close)
if np.isclose(m, m_sklearn) and np.isclose(b, b_sklearn):
    print("\nVerification: The manual and scikit-learn parameters match.")
else:
    print("\nVerification: The manual and scikit-learn parameters do not match.")

# Plot the regression line from scikit-learn for comparison
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, Y_pred_manual, color='red', label='Manual Regression Line')

# Calculate the predicted Y values using scikit-learn's model
Y_pred_sklearn = model.predict(X_reshaped)
plt.plot(X, Y_pred_sklearn, color='green', linestyle='--', label='Scikit-learn Regression Line')

plt.title('Linear Regression: Manual vs Scikit-learn')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (Y)')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Calculate and display the R-squared value of the regression
# Using the manual predictions
r2_manual = r2_score(Y, Y_pred_manual)
print(f"\nManual Regression R-squared: {r2_manual}")

# Using scikit-learn's model
r2_sklearn = model.score(X_reshaped, Y)
print(f"Scikit-learn Regression R-squared: {r2_sklearn}")

# Optionally, display which R-squared is higher (they should be the same or very close)
if np.isclose(r2_manual, r2_sklearn):
    print("\nVerification: The R-squared values from manual calculation and scikit-learn match.")
else:
    print("\nVerification: The R-squared values from manual calculation and scikit-learn do not match.")
