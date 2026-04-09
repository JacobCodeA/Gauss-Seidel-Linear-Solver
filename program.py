import os
import numpy as np


def get_int(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = int(input(prompt))

            if min_value is not None and value < min_value:
                print(f"Value must be >= {min_value}. Try again.")
            elif max_value is not None and value > max_value:
                print(f"Value must be <= {max_value}. Try again.")
            else:
                return value
        except ValueError:
            print("Invalid integer value. Try again.")


def get_float(prompt, allow_zero=True):
    while True:
        try:
            value = float(input(prompt))

            if not allow_zero and value == 0:
                print("Zero is not allowed here. Try again.")
            else:
                return value
        except ValueError:
            print("Invalid floating-point value. Try again.")


def get_choice(prompt, allowed):
    allowed = set(allowed)
    while True:
        value = input(prompt).strip()
        if value in allowed:
            return value
        print(f"Choose one of the options: {sorted(allowed)}")


def get_txt_file(prompt, must_exist=False):
    while True:
        filename = input(prompt).strip()

        if not filename.endswith(".txt"):
            print("The file must have a .txt extension.")
        elif must_exist and not os.path.isfile(filename):
            print("File not found. Try again.")
        else:
            return filename


class LinearSystem:
    def __init__(self, A, b):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float).reshape(-1)

        if self.A.ndim != 2:
            raise ValueError("Matrix A must be two-dimensional.")

        rows, cols = self.A.shape
        if rows != cols:
            raise ValueError("Matrix A must be square.")

        if self.b.shape[0] != rows:
            raise ValueError("Vector b size must match the matrix size.")

        self.n = rows

    @classmethod
    def from_manual(cls):
        n = get_int("Enter the size of the system (n): ", min_value=1)

        A = np.zeros((n, n), dtype=float)
        b = np.zeros(n, dtype=float)

        print("\nEnter matrix A:")
        for i in range(n):
            for j in range(n):
                A[i, j] = get_float(f"A[{i}][{j}] = ")

        print("\nEnter vector b:")
        for i in range(n):
            b[i] = get_float(f"b[{i}] = ")

        return cls(A, b)

    @classmethod
    def from_file(cls, filename):
        if not filename.endswith(".txt"):
            raise ValueError("Input file must have a .txt extension.")

        if not os.path.isfile(filename):
            raise FileNotFoundError("Input file was not found.")

        with open(filename, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        if not lines:
            raise ValueError("The input file is empty.")

        n = int(lines[0])
        if n < 1:
            raise ValueError("System size must be positive.")

        if len(lines) - 1 != n:
            raise ValueError(f"Expected exactly {n} equation rows after the first line.")

        data = []
        for i in range(1, n + 1):
            values = list(map(float, lines[i].split()))
            if len(values) != n + 1:
                raise ValueError(
                    f"Row {i} must contain exactly {n + 1} numbers: {n} coefficients and 1 value from vector b."
                )
            data.append(values)

        data = np.array(data, dtype=float)
        A = data[:, :n]
        b = data[:, n]

        return cls(A, b)

    def save_to_file(self, filename):
        with open(filename, "w", encoding="utf-8") as file:
            file.write(f"{self.n}\n")
            for i in range(self.n):
                row = list(self.A[i]) + [self.b[i]]
                file.write(" ".join(f"{value:.10f}" for value in row) + "\n")

    def print_system(self):
        print("\nMatrix A:")
        print(self.A)
        print("\nVector b:")
        print(self.b)

    def is_diagonally_dominant(self):
        diagonal = np.abs(np.diag(self.A))
        row_sum = np.sum(np.abs(self.A), axis=1) - diagonal
        return np.all(diagonal > row_sum)

    def is_symmetric(self):
        return np.allclose(self.A, self.A.T, atol=1e-12)

    def is_positive_definite(self):
        try:
            np.linalg.cholesky(self.A)
            return True
        except np.linalg.LinAlgError:
            return False

    def convergence_info(self):
        if self.is_diagonally_dominant():
            return "diagonal_dominance"
        if self.is_symmetric() and self.is_positive_definite():
            return "spd"
        return "unknown"

    def consistency_info(self):
        augmented = np.column_stack((self.A, self.b))
        rank_a = np.linalg.matrix_rank(self.A)
        rank_ab = np.linalg.matrix_rank(augmented)

        if rank_a < rank_ab:
            return "inconsistent"
        if rank_a < self.n:
            return "infinitely_many"
        return "unique"

    def fix_diagonal(self, tol=1e-12):
        for i in range(self.n):
            pivot_row = i
            pivot_value = abs(self.A[i, i])

            for j in range(i + 1, self.n):
                candidate_value = abs(self.A[j, i])
                if candidate_value > pivot_value:
                    pivot_value = candidate_value
                    pivot_row = j

            if pivot_value <= tol:
                raise ValueError(f"Cannot place a non-zero pivot on the diagonal at row {i}.")

            if pivot_row != i:
                self.A[[i, pivot_row]] = self.A[[pivot_row, i]]
                self.b[[i, pivot_row]] = self.b[[pivot_row, i]]
                print(f"Swapped rows {i} and {pivot_row} to improve the diagonal.")

    def gauss_seidel_step(self, x, b_vector):
        x_new = x.copy()

        for i in range(self.n):
            if abs(self.A[i, i]) < 1e-12:
                raise ValueError(f"Near-zero diagonal element at row {i}.")

            left_part = np.dot(self.A[i, :i], x_new[:i])
            right_part = np.dot(self.A[i, i + 1:], x[i + 1:])
            x_new[i] = (b_vector[i] - left_part - right_part) / self.A[i, i]

        return x_new

    def gauss_seidel(self, iterations=None, eps=None, max_iter=1000):
        if (iterations is None and eps is None) or (iterations is not None and eps is not None):
            raise ValueError("Choose exactly one stopping criterion: iterations or eps.")

        x = np.zeros(self.n, dtype=float)
        b_vector = np.array(self.b, dtype=float)

        if eps is not None:
            if eps <= 0:
                raise ValueError("Epsilon must be positive.")

            print("\n--- Gauss-Seidel method (epsilon stopping criterion) ---")
            for iteration in range(1, max_iter + 1):
                x_new = self.gauss_seidel_step(x, b_vector)
                print(f"Iteration {iteration}: x = {x_new}")

                if np.linalg.norm(x_new - x, ord=np.inf) < eps:
                    print(f"\nConverged after {iteration} iterations.")
                    return x_new

                x = x_new

            print("\nWarning: maximum number of iterations reached without convergence.")
            return x

        if iterations <= 0:
            raise ValueError("Iteration count must be positive.")

        print("\n--- Gauss-Seidel method (fixed iteration count) ---")
        for iteration in range(1, iterations + 1):
            x = self.gauss_seidel_step(x, b_vector)
            print(f"Iteration {iteration}: x = {x}")

        return x


def start():
    print("Gauss-Seidel solver for a system of linear equations\n")

    input_mode = get_choice(
        "Choose input mode:\n1. Manual input\n2. Load from file\nYour choice: ",
        {"1", "2"}
    )

    try:
        if input_mode == "1":
            system = LinearSystem.from_manual()

            save_choice = get_choice(
                "\nDo you want to save the system to a .txt file? (y/n): ",
                {"y", "n"}
            )
            if save_choice == "y":
                output_file = get_txt_file("Enter output file name (.txt): ")
                system.save_to_file(output_file)
                print("System saved successfully.")
        else:
            input_file = get_txt_file("Enter input file name (.txt): ", must_exist=True)
            system = LinearSystem.from_file(input_file)

    except Exception as error:
        print(f"Input error: {error}")
        return

    system.print_system()

    consistency = system.consistency_info()
    if consistency == "inconsistent":
        print("\nThe system is inconsistent. No solution exists.")
        return
    if consistency == "infinitely_many":
        print("\nThe system has infinitely many solutions.")
        return

    try:
        system.fix_diagonal()
    except Exception as error:
        print(f"\nDiagonal correction failed: {error}")
        return

    print("\nSystem after diagonal correction:")
    system.print_system()

    convergence = system.convergence_info()
    if convergence == "diagonal_dominance":
        print("\nConvergence check: the matrix is diagonally dominant.")
    elif convergence == "spd":
        print("\nConvergence check: the matrix is symmetric positive definite.")
    else:
        print("\nWarning: no convergence guarantee was detected.")
        continue_choice = get_choice("Do you want to continue anyway? (y/n): ", {"y", "n"})
        if continue_choice == "n":
            return

    print("\nChoose the stopping criterion:")
    stopping_mode = get_choice("1. Epsilon\n2. Iteration limit\nYour choice: ", {"1", "2"})

    try:
        if stopping_mode == "1":
            eps = get_float("Enter epsilon: ", allow_zero=False)
            max_iter = get_int("Enter the maximum number of iterations: ", min_value=1)
            result = system.gauss_seidel(eps=eps, max_iter=max_iter)
        else:
            iterations = get_int("Enter the number of iterations: ", min_value=1)
            result = system.gauss_seidel(iterations=iterations)

        print("\nSolution (Gauss-Seidel):")
        print(result)

    except Exception as error:
        print(f"Calculation error: {error}")


if __name__ == "__main__":
    start()