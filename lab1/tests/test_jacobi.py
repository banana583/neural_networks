"""Lab 1 regression tests; run with python -m unittest discover -s tests -v."""

import copy
import json
import math
import random
import subprocess
import sys
import unittest
from pathlib import Path

from jacobi_eigen import (
    ConvergenceError,
    EigenResult,
    InvalidMatrixError,
    InvalidParameterError,
    NumericalError,
    jacobi_eigendecomposition,
)


def assert_decomposition(testcase, a, answer, *, atol=1e-9):
    """Check A*V=V*D, V^T*V=I and ascending eigenvalues independently."""
    n = len(a)
    vals = answer.eigenvalues
    v = answer.eigenvectors
    testcase.assertEqual(len(vals), n)
    testcase.assertEqual(len(v), n)
    testcase.assertTrue(all(len(row) == n for row in v))
    testcase.assertTrue(all(vals[j] <= vals[j + 1] for j in range(n - 1)))
    for col in range(n):
        for row in range(n):
            av = math.fsum(a[row][k] * v[k][col] for k in range(n))
            vd = vals[col] * v[row][col]
            testcase.assertAlmostEqual(av, vd, delta=atol)
    for p in range(n):
        for q in range(n):
            dot = math.fsum(v[k][p] * v[k][q] for k in range(n))
            testcase.assertAlmostEqual(dot, float(p == q), delta=atol)
    testcase.assertAlmostEqual(sum(vals), sum(a[i][i] for i in range(n)), delta=atol)
    testcase.assertGreaterEqual(answer.rotations, 0)
    testcase.assertGreaterEqual(answer.sweeps, 0)
    testcase.assertGreaterEqual(answer.off_diagonal_norm, 0)


class JacobiValidCases(unittest.TestCase):
    def test_simple_2_by_2(self):
        a = [[2.0, 1.0], [1.0, 2.0]]
        result = jacobi_eigendecomposition(a)
        self.assertAlmostEqual(result.eigenvalues[0], 1.0, places=12)
        self.assertAlmostEqual(result.eigenvalues[1], 3.0, places=12)
        assert_decomposition(self, a, result)
        self.assertEqual(result.rotations, 1)

    def test_simple_3_by_3(self):
        a = [[4, 1, 1], [1, 3, 0], [1, 0, 2]]
        result = jacobi_eigendecomposition(a)
        expected = [1.467911113762044, 2.652703644666139, 4.879385241571816]
        for actual, target in zip(result.eigenvalues, expected):
            self.assertAlmostEqual(actual, target, places=10)
        assert_decomposition(self, a, result)

    def test_example_from_pdf(self):
        a = [[1 + 3 * (i + j) for j in range(5)] for i in range(5)]
        result = jacobi_eigendecomposition(a)
        self.assertAlmostEqual(result.eigenvalues[0], -6.31043674065006, places=9)
        self.assertAlmostEqual(result.eigenvalues[-1], 71.31043674065006, places=9)
        for ev in result.eigenvalues[1:4]:
            self.assertAlmostEqual(ev, 0, delta=1e-10)
        assert_decomposition(self, a, result)

    def test_singleton(self):
        result = jacobi_eigendecomposition([[-7]])
        self.assertEqual(result.eigenvalues, (-7.0,))
        self.assertEqual(result.eigenvectors, ((1.0,),))
        self.assertEqual(result.sweeps, 0)
        self.assertEqual(result.rotations, 0)

    def test_zero_matrix(self):
        result = jacobi_eigendecomposition([[0, 0], [0, 0]])
        self.assertEqual(result.eigenvalues, (0.0, 0.0))
        self.assertEqual(result.off_diagonal_norm, 0.0)
        assert_decomposition(self, [[0, 0], [0, 0]], result)

    def test_diagonal_unsorted(self):
        a = [[5, 0, 0], [0, -2, 0], [0, 0, 1]]
        result = jacobi_eigendecomposition(a)
        self.assertEqual(result.eigenvalues, (-2.0, 1.0, 5.0))
        self.assertEqual(result.rotations, 0)
        assert_decomposition(self, a, result)

    def test_no_sort_retains_order(self):
        result = jacobi_eigendecomposition([[5, 0], [0, -2]], sort=False)
        self.assertEqual(result.eigenvalues, (5.0, -2.0))
        self.assertEqual(result.eigenvectors, ((1.0, 0.0), (0.0, 1.0)))

    def test_repeated_eigenvalues(self):
        a = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        result = jacobi_eigendecomposition(a)
        self.assertEqual(result.eigenvalues, (2.0, 2.0, 2.0))
        assert_decomposition(self, a, result)

    def test_indefinite_negative_offdiagonal(self):
        a = [[-2.0, -4.0], [-4.0, 7.0]]
        result = jacobi_eigendecomposition(a)
        assert_decomposition(self, a, result)
        self.assertLess(result.eigenvalues[0], 0)
        self.assertGreater(result.eigenvalues[1], 0)

    def test_dense_deterministic_8(self):
        rng = random.Random(421)
        n = 8
        a = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                a[i][j] = a[j][i] = rng.uniform(-3, 3)
        result = jacobi_eigendecomposition(a)
        assert_decomposition(self, a, result, atol=1e-9)
        self.assertLess(result.off_diagonal_norm**2, 1e-24)

    def test_12_by_12_random(self):
        rng = random.Random(220)
        n = 12
        a = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                a[i][j] = a[j][i] = rng.uniform(-1, 1)
        result = jacobi_eigendecomposition(a)
        assert_decomposition(self, a, result, atol=1e-9)

    def test_large_values_no_intermediate_overflow(self):
        a = [[1e200, 1e200], [1e200, 1e200]]
        result = jacobi_eigendecomposition(a)
        self.assertAlmostEqual(result.eigenvalues[0], 0, delta=1e186)
        self.assertAlmostEqual(result.eigenvalues[1], 2e200, delta=1e186)
        self.assertTrue(math.isfinite(result.off_diagonal_norm))

    def test_small_values(self):
        a = [[1e-200, 0], [0, 2e-200]]
        result = jacobi_eigendecomposition(a)
        self.assertEqual(result.eigenvalues, (1e-200, 2e-200))

    def test_very_small_epsilon_with_exact_diagonal(self):
        a = [[1e200, 0.0], [0.0, -1e200]]
        result = jacobi_eigendecomposition(a, epsilon=5e-324)
        self.assertEqual(result.sweeps, 0)

    def test_diagonal_matrix_with_extreme_dynamic_range(self):
        # An exactly diagonal matrix must not lose its small eigenvalue.
        a = [[1e308, 0.0], [0.0, 1e-20]]
        result = jacobi_eigendecomposition(a)
        self.assertEqual(result.eigenvalues, (1e-20, 1e308))
        self.assertEqual(result.rotations, 0)
        self.assertEqual(result.off_diagonal_norm, 0.0)
        self.assertEqual(result.eigenvectors, ((0.0, 1.0), (1.0, 0.0)))

    def test_input_is_not_mutated(self):
        a = [[2.0, 1.0], [1.0, 3.0]]
        before = copy.deepcopy(a)
        jacobi_eigendecomposition(a)
        self.assertEqual(a, before)

    def test_tuple_input_and_result_immutable(self):
        result = jacobi_eigendecomposition(((2, 1), (1, 2)))
        self.assertIsInstance(result, EigenResult)
        with self.assertRaises((AttributeError, TypeError)):
            result.eigenvalues = (1, 3)

    def test_challenging_small_offdiagonal(self):
        a = [[1.0, 1e-13], [1e-13, 2.0]]
        result = jacobi_eigendecomposition(a, epsilon=1e-30)
        self.assertLess(result.off_diagonal_norm**2, 1e-30)
        assert_decomposition(self, a, result)


class JacobiBadInputs(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(InvalidMatrixError):
            jacobi_eigendecomposition([])

    def test_nonsquare(self):
        with self.assertRaises(InvalidMatrixError):
            jacobi_eigendecomposition([[1, 2], [2, 3], [4, 5]])

    def test_non_symmetric(self):
        with self.assertRaises(InvalidMatrixError):
            jacobi_eigendecomposition([[1, 2], [3, 1]])

    def test_not_a_sequence(self):
        with self.assertRaises(InvalidMatrixError):
            jacobi_eigendecomposition(5)

    def test_invalid_element(self):
        for a in ([[1, "a"], ["a", 1]], [[1, True], [True, 1]]):
            with self.subTest(a=a), self.assertRaises(InvalidMatrixError):
                jacobi_eigendecomposition(a)

    def test_nan_and_infinity(self):
        for x in (math.nan, math.inf, -math.inf):
            with self.subTest(x=x), self.assertRaises(InvalidMatrixError):
                jacobi_eigendecomposition([[x]])

    def test_nonfinite_epsilon(self):
        for eps in (0, -1, math.inf, math.nan, True, "1e-10"):
            with self.subTest(eps=eps), self.assertRaises(InvalidParameterError):
                jacobi_eigendecomposition([[1]], epsilon=eps)

    def test_invalid_max_sweeps(self):
        for sweeps in (0, -1, 1.5, False, "12"):
            with self.subTest(sweeps=sweeps), self.assertRaises(InvalidParameterError):
                jacobi_eigendecomposition([[1]], max_sweeps=sweeps)

    def test_invalid_sort(self):
        with self.assertRaises(InvalidParameterError):
            jacobi_eigendecomposition([[1]], sort="yes")

    def test_limited_iterations_fail_predictably(self):
        a = [[4, 1, 1], [1, 3, 0], [1, 0, 2]]
        with self.assertRaises(ConvergenceError):
            jacobi_eigendecomposition(a, max_sweeps=1, epsilon=1e-30)

    def test_scaling_must_not_silently_erase_an_offdiagonal(self):
        a = [[1e308, 1e-20], [1e-20, -1e308]]
        with self.assertRaises(NumericalError):
            jacobi_eigendecomposition(a, epsilon=1e-100)

    def test_scaling_must_not_silently_erase_a_diagonal_entry(self):
        a = [[1e308, 1.0, 0], [1.0, 0.0, 0], [0, 0, 1e-20]]
        with self.assertRaises(NumericalError):
            jacobi_eigendecomposition(a, epsilon=1e-100)

    def test_eigenvalue_overflow_is_controlled(self):
        with self.assertRaises(NumericalError):
            jacobi_eigendecomposition([[1e308, 1e308], [1e308, 1e308]])


class JacobiCommandLine(unittest.TestCase):
    ROOT = Path(__file__).resolve().parent.parent

    def run_cli(self, *args, input_text=None):
        return subprocess.run(
            [sys.executable, "-m", "jacobi_eigen", *args],
            cwd=self.ROOT, input=input_text, capture_output=True, text=True,
            check=False,
        )

    def test_cli_example_json(self):
        process = self.run_cli("examples/matrix.json")
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(len(result["eigenvalues"]), 3)
        self.assertEqual(len(result["eigenvectors"]), 3)

    def test_cli_stdin(self):
        process = self.run_cli("-", input_text='{"matrix": [[2,1],[1,2]]}')
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["eigenvalues"], [1.0, 3.0])

    def test_cli_invalid_matrix_returns_controlled_error(self):
        process = self.run_cli("-", input_text='{"matrix": [[1,2],[3,4]]}')
        self.assertEqual(process.returncode, 2)
        self.assertIn("Error:", process.stderr)
        self.assertNotIn("Traceback", process.stderr)

    def test_cli_invalid_json_returns_controlled_error(self):
        process = self.run_cli("-", input_text="not json")
        self.assertEqual(process.returncode, 2)
        self.assertIn("Error:", process.stderr)
        self.assertNotIn("Traceback", process.stderr)


if __name__ == "__main__":
    unittest.main()
