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