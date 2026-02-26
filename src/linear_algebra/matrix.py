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