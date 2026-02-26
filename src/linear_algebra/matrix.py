import math
class Matrix:
    def __init__(self, data):
        """
        data: list of lists (2D list)
        Example:
        [[1, 2],
         [3, 4]]
        """
        if not data or not all(len(row) == len(data[0]) for row in data):
            raise ValueError("All rows must have the same number of columns.")

        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

    def shape(self):
        return (self.rows, self.cols)

    def __repr__(self):
        return f"Matrix({self.data})"
    
    def __add__(self, other):
        if self.shape() != other.shape():
            raise ValueError("Matrix dimensions must match for addition.")

        result = [
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]

        return Matrix(result)
    def __mul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Matrix dimensions not compatible for multiplication.")

        result = [
            [0 for _ in range(other.cols)]
            for _ in range(self.rows)
        ]

        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.data[i][k] * other.data[k][j]

        return Matrix(result)
    def transpose(self):
        result = [
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ]
        return Matrix(result)
    def determinant(self):
        if self.rows != self.cols:
            raise ValueError("Determinant only defined for square matrices.")

        # 2x2 base case
        if self.rows == 2:
            return (
                self.data[0][0] * self.data[1][1]
                - self.data[0][1] * self.data[1][0]
            )

        # 1x1 case
        if self.rows == 1:
            return self.data[0][0]

        # Recursive case (Laplace expansion)
        det = 0
        for col in range(self.cols):
            submatrix = [
                [self.data[i][j] for j in range(self.cols) if j != col]
                for i in range(1, self.rows)
            ]
            sign = (-1) ** col
            det += sign * self.data[0][col] * Matrix(submatrix).determinant()

        return det
    def inverse(self):
        if self.rows != self.cols:
            raise ValueError("Inverse only defined for square matrices.")

        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")

        # 2x2 shortcut
        if self.rows == 2:
            return Matrix([
                [ self.data[1][1] / det, -self.data[0][1] / det],
                [-self.data[1][0] / det,  self.data[0][0] / det]
            ])

        # General case
        cofactors = []
        for i in range(self.rows):
            cofactor_row = []
            for j in range(self.cols):
                minor = [
                    [self.data[r][c] for c in range(self.cols) if c != j]
                    for r in range(self.rows) if r != i
                ]
                sign = (-1) ** (i + j)
                cofactor_row.append(sign * Matrix(minor).determinant())
            cofactors.append(cofactor_row)

        cofactor_matrix = Matrix(cofactors)
        adjugate = cofactor_matrix.transpose()

        inverse_data = [
            [adjugate.data[i][j] / det for j in range(self.cols)]
            for i in range(self.rows)
        ]

        return Matrix(inverse_data)
    def eigenvalues(self):
        if self.rows != 2 or self.cols != 2:
            raise ValueError("Eigenvalue method currently supports only 2x2 matrices.")

        a = self.data[0][0]
        b = self.data[0][1]
        c = self.data[1][0]
        d = self.data[1][1]

        trace = a + d
        det = self.determinant()

        discriminant = trace**2 - 4 * det

        if discriminant < 0:
            raise ValueError("Complex eigenvalues not supported yet.")

        sqrt_disc = math.sqrt(discriminant)

        lambda1 = (trace + sqrt_disc) / 2
        lambda2 = (trace - sqrt_disc) / 2

        return (lambda1, lambda2)