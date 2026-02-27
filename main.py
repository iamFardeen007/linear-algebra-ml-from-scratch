from src.linear_algebra.matrix import Matrix

A = Matrix([[1, 2], [3, 4]])
print(A)
print("Shape:", A.shape())
B = Matrix([[5, 6], [7, 8]])
C = A + B
print("Addition Result:", C)
D = A * B
print("Multiplication Result:", D)
T = A.transpose()
print("Transpose:", T)
print("Determinant:", A.determinant())
inv_A = A.inverse()
print("Inverse:", inv_A)
eigen_vals = A.eigenvalues()
print("Eigenvalues:", eigen_vals)
import numpy as np
from src.ml.linear_regression import LinearRegression

# Simple dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 5, 7, 9, 11])  # y = 2x + 1

model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X, y)

predictions = model.predict(X)

print("Learned parameters:", model.theta)
print("Predictions:", predictions)