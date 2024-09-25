import numpy as np
import matplotlib.pyplot as plt

# Define the non-convex function f(x, y)
def f(x, y):
    return x**2 + y**2 - x*y + 10

# Define the gradient of the function
def gradient(x, y):
    df_dx = 2*x - y  # Partial derivative w.r.t x
    df_dy = 2*y - x  # Partial derivative w.r.t y
    return np.array([df_dx, df_dy])

# Gradient Descent Function
def gradient_descent(starting_point, learning_rate, max_iter=100, tol=1e-6):
    point = np.array(starting_point)
    history = [point.copy()]

    for i in range(max_iter):
        grad = gradient(point[0], point[1])
        new_point = point - learning_rate * grad
        
        history.append(new_point.copy())
        
        # Check for convergence
        if np.linalg.norm(new_point - point) < tol:
            break
            
        point = new_point

    return point, history

# Parameters
initial_points = [(3, 3), (-3, -3), (0, 0), (1, -1)]
learning_rates = [0.1, 0.01, 0.001]
results = {}

# Run Gradient Descent for multiple initial points and learning rates
for lr in learning_rates:
    for initial in initial_points:
        final_point, history = gradient_descent(initial, lr)
        results[(initial, lr)] = (final_point, history)

# Plotting the function and the paths taken
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(12, 8))
contour = plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.title('Gradient Descent on Non-Convex Function')
plt.xlabel('x')
plt.ylabel('y')

# Plotting the paths taken
for (initial, lr), (final_point, history) in results.items():
    history = np.array(history)
    plt.plot(history[:, 0], history[:, 1], marker='o', label=f'Init: {initial}, lr: {lr}')

plt.legend()
plt.show()

# Print final points
for (initial, lr), (final_point, history) in results.items():
    print(f'Initial point: {initial}, Learning rate: {lr}, Final point: {final_point}, f(x,y): {f(final_point[0], final_point[1])}')
