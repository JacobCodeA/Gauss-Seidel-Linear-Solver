import numpy as np

def get_int(prompt, min_value=None, max_value=None):
    while True:
        try:
            v = int(input(prompt))
            valid = True

            if min_value is not None and v < min_value:
                print(f"Value must be >= {min_value}. Try again.")
                valid = False

            if max_value is not None and v > max_value:
                print(f"Value must be <= {max_value}. Try again.")
                valid = False

            if valid:
                return v

        except ValueError:
            print("Invalid integer value. Try again.")


def get_float(prompt, allow_zero=True):
    while True:
        try:
            v = float(input(prompt))

            if not allow_zero and v == 0:
                print("Zero is not allowed here. Try again.")
            else:
                return v

        except ValueError:
            print("Invalid floating-point value. Try again.")


def get_choice(prompt, allowed):
    allowed = set(allowed)
    while True:
        v = input(prompt).strip()
        if v in allowed:
            return v
        print(f"Choose one of the options: {sorted(allowed)}")


def get_txt_file():
    while True:
        file = input("Enter file name (.txt): ").strip()
        if file.endswith(".txt"):
            return file
        print("File must have a .txt extension")


def get_vector(n):
    vector = []
    print("Enter values for vector b:")
    for i in range(n):
        val = get_float(f"b[{i}] = ")
        vector.append(val)
    return vector

class Matrix:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.numbers = np.zeros((height, width), dtype=float)

        for i in range(height):
            for j in range(width):
                self.numbers[i, j] = get_float(f"Input number for position [{j}][{i}]: ")

    @classmethod
    def from_file(cls, file=None):
        if not file or not file.endswith(".txt"):
            print("Wrong file format")
            exit(1)

        try:
            numbers = np.loadtxt(file, dtype=float)

            if numbers.ndim != 2:
                print("Matrix is not properly structured")
                exit(1)

            height, width = numbers.shape

            obj = cls.__new__(cls)
            obj.numbers = numbers
            obj.width = width
            obj.height = height

            return obj

        except FileNotFoundError:
            print("File is not found")
            exit(1)

    def save_to_file(self, file):
        np.savetxt(file, self.numbers, fmt='%.6f')

    def change_value(self, row, col, value):
        self.numbers[row, col] = value

    def print_matrix(self):
        print(self.numbers)

    def is_diagonally_dominant(self):
        diagonal = np.abs(np.diag(self.numbers))
        row_sum = np.sum(np.abs(self.numbers), axis=1) - diagonal
        return np.all(diagonal > row_sum)

    def is_symmetric(self):
        return np.allclose(self.numbers, self.numbers.T)

    def _determinant(self):
        if self.height == self.width:
            return np.linalg.det(self.numbers)
        else:
            print("Determinant apply only to square matrix!")

    def is_positive_definite(self):
        try:
            np.linalg.cholesky(self.numbers)
            return True
        except np.linalg.LinAlgError:
            return False

    def is_convergent(self):
        if self.is_diagonally_dominant():
            print("Matrix is diagonally dominant and convergent!")
            return True
        elif self.is_symmetric() and self.is_positive_definite():
            print("Matrix is symmetric and positive definite, convergent!")
            return True
        else:
            print("Matrix is not convergent!")
            return False

    def is_consistent(self, b_vector):
        a = self.numbers
        b_vector = np.array(b_vector, dtype=float).reshape(-1, 1)
        a_b_vector = np.hstack([a, b_vector])

        rank_a = np.linalg.matrix_rank(a)
        rank_ab = np.linalg.matrix_rank(a_b_vector)
        n = self.width

        if rank_a < rank_ab:
            print("Matrix is not consistent — no solutions")
            return False
        elif rank_a < n:
            print("Matrix is indefinite — infinite number of solutions")
            return False
        else:
            print("Matrix is consistent — one solution")
            return True

    def gauss_seidel_step(self, x, b_vector):
        n = self.height
        x_new = x.copy()

        for i in range(n):
            if self.numbers[i, i] == 0:
                print(f"Zero on diagonal at row {i} — cannot divide")
                exit(1)

            sum1 = np.dot(self.numbers[i, :i], x_new[:i])
            sum2 = np.dot(self.numbers[i, i + 1:], x[i + 1:])
            x_new[i] = (b_vector[i] - sum1 - sum2) / self.numbers[i, i]

        return x_new

    def gauss_seidel(self, b_vector, itr=None, eps=None):
        if self.width != self.height:
            print("Matrix must be square!")
            exit(1)
        x = np.zeros(self.height)
        b_vector = np.array(b_vector, dtype=float)

        if eps is not None:
            print("\n--- Gauss-Seidel (epsilon mode) ---")
            iteration = 1

            while True:
                x_new = self.gauss_seidel_step(x, b_vector)

                print(f"Iteration {iteration}: x = {x_new}")

                if np.linalg.norm(x_new - x, ord=np.inf) < eps:
                    print(f"\nConverged after {iteration} iterations.")
                    return x_new

                x = x_new
                iteration += 1

        if itr is not None:
            print("\n--- Gauss-Seidel (iteration limit mode) ---")

            for iteration in range(1, itr + 1):
                x_new = self.gauss_seidel_step(x, b_vector)

                print(f"Iteration {iteration}: x = {x_new}")

                x = x_new

            print(f"\nFinished after {itr} iterations.")
            return x

def start():
    opt = get_choice("1. Input manually matrix value\n2. Load matrix from file\nChoose option: ",{"1", "2"})

    if opt == "1":
        width = get_int("Input width of matrix : ", min_value=1)
        height = get_int("Input height of matrix : ", min_value=1)
        m = Matrix(width, height)

    else:
        file = get_txt_file()
        m = Matrix.from_file(file)

    print("-----------------------------------------------------------------------\n")

    m.print_matrix()

    if not m.is_convergent():
        exit(1)

    print("------------------------ b vector construction ------------------------")

    b = get_vector(m.height)

    if not m.is_consistent(b):
        exit(1)

    print("---------------------- Calculation end condition ----------------------")
    condition = get_choice("1.Epsilon (precision)\n2.Iteration limit\nInput calculation end condition : ", {"1","2"})

    if condition == "1":
        eps = get_float("\nInput Epsilon value : ")
        result = m.gauss_seidel(b, eps = eps)
    else:
        itr = get_int("\nInput iteration limit : ")
        result = m.gauss_seidel(b, itr = itr)


    print("\nSolution (Gauss-Seidel):")
    print(result)

if __name__ == '__main__':
    start()