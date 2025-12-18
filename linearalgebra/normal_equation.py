import numpy as np

# Data
X = np.array([
    [1.000, 2.345, 0.628, 0.982],
    [1.000, 2.334, -0.481, -0.607],
    [1.000, 1.784, 1.155, 1.855],
    [1.000, 0.933, 0.157, 1.855]
])

y = np.array([4.526, 3.585, 3.521, 3.413]).reshape(-1,1)

# Normal Equation
X_T = X.T
X_T_X = X_T @ X
X_T_X_inv = np.linalg.inv(X_T_X)
X_T_y = X_T @ y
w = X_T_X_inv @ X_T_y

print("Weights w:", w.flatten())
print("\nPredictions:", X @ w)
print("Actual y:", y.flatten())