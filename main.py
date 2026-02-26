from src.linear_algebra.matrix import Matrix

A = Matrix([[1, 2], [3, 4]])
print(A)
print("Shape:", A.shape())
B = Matrix([[5, 6], [7, 8]])
C = A + B
print("Addition Result:", C)
D = A * B
print("Multiplication Result:", D)