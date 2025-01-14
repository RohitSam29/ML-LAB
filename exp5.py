import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 - 3*x + 20

def df(x):
    return 2*x - 3

# Gradient Descent Function
def gradient_descent(start_point, learning_rate, iterations):
    x = start_point
    path = [x]  # To store the path of x values
    for i in range(iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        path.append(x)
    return path

# Initial Data Points for Plotting the Function
x_values = np.linspace(-4, 12, 400)
y_values = f(x_values)

# Gradient Descent Parameters for Different Runs
params = [
    {'start_point': 10, 'learning_rate': 0.01, 'iterations': 500, 'label': 'L=0.01, N=500'},
    {'start_point': 10, 'learning_rate': 0.1, 'iterations': 100, 'label': 'L=0.1, N=100'},
    {'start_point': 10, 'learning_rate': 1.0, 'iterations': 100, 'label': 'L=1.0, N=100'},
    {'start_point': 10, 'learning_rate': 0.5, 'iterations': 100, 'label': 'L=0.5, N=100'},
    {'start_point': 10, 'learning_rate': 0.75, 'iterations': 100, 'label': 'L=0.75, N=100'}
]

# Plotting the Function
plt.figure(figsize=(12, 8))
plt.plot(x_values, y_values, 'b-', label='Function y = x² - 3x + 20')
plt.title('Gradient Descent on y = x² - 3x + 20')
plt.xlabel('x')
plt.ylabel('y')

# Perform Gradient Descent for Each Parameter Set and Plot the Path
for param in params:
    path = gradient_descent(param['start_point'], param['learning_rate'], param['iterations'])
    path_x = path
    path_y = f(np.array(path_x))
    plt.plot(path_x, path_y, marker='x', linestyle='--', label=f"Path {param['label']}")

    # Optionally, plot the final point
    plt.plot(path_x[-1], path_y[-1], marker='o', markersize=8)

# Highlight the Actual Minimum
minimum_x = 1.5  # From calculus, the minimum of y = x^2 - 3x + 20 is at x = 1.5
minimum_y = f(minimum_x)
plt.plot(minimum_x, minimum_y, 'r*', markersize=15, label='Actual Minimum')

plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig('gradient_descent_plot.png')
print("Plot saved as 'gradient_descent_plot.png'.")

plt.show()

