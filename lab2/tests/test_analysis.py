import itertools
import math
import unittest

import numpy as np
from random_matrix_lab import (
    analyze_ensemble, central_gap_index, exact_binary_2x2_distribution,
    generate_ensemble, ks_distance, neighbor_spacings, normalize_spacings,
    wigner_cdf, wigner_density,
)


class Lab2Tests(unittest.TestCase):
    def test_central_gap_indices(self):
        self.assertEqual([central_gap_index(n) for n in (2, 4, 16)], [0, 1, 7])

    def test_invalid_size(self):
        for size in (0, 1, -1, 2.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                central_gap_index(size)

    def test_generates_correct_shapes_and_symmetric_matrices(self):
        for dist in ("normal", "rademacher"):
            for n in (2, 4, 16):
                with self.subTest(dist=dist, n=n):
                    result = generate_ensemble(n, 24, dist, np.random.default_rng(123))
                    self.assertEqual(result.shape, (24, n, n))
                    self.assertTrue(np.array_equal(result, result.swapaxes(-1, -2)))

    def test_reproducible_generation(self):
        for dist in ("normal", "rademacher"):
            self.assertTrue(np.array_equal(generate_ensemble(4, 20, dist, np.random.default_rng(67)),
                                           generate_ensemble(4, 20, dist, np.random.default_rng(67))))

    def test_normal_gaussian_symmetrization_variances(self):
        matrices = generate_ensemble(3, 100000, "normal", np.random.default_rng(11))
        self.assertAlmostEqual(float(np.var(matrices[:, 0, 0])), 4, delta=.09)
        self.assertAlmostEqual(float(np.var(matrices[:, 0, 1])), 2, delta=.06)
        self.assertAlmostEqual(float(np.mean(matrices[:, 0, 0])), 0, delta=.03)

    def test_binary_support(self):
        matrices = generate_ensemble(4, 1000, "rademacher", np.random.default_rng(8))
        self.assertTrue(np.isin(matrices, (-2, 0, 2)).all())
        self.assertTrue(np.isin(np.diagonal(matrices, axis1=-2, axis2=-1), (-2, 2)).all())

    def test_exact_2_by_2_eigenvalues_and_gap(self):
        matrices = np.array([[[2, 1], [1, 2]], [[3, -1], [-1, 0]]], dtype=float)
        gaps = neighbor_spacings(matrices, 0)
        self.assertTrue(np.allclose(gaps, [2, math.sqrt(13)]))

    def test_2x2_exhaustive_rademacher_probabilities(self):
        gaps = []
        for vals in itertools.product((-1.0, 1.0), repeat=4):
            x = np.array(vals).reshape(2, 2)
            a = x + x.T
            gaps.append(float(np.linalg.eigvalsh(a)[1] - np.linalg.eigvalsh(a)[0]))
        expected_s, normalized_s, probabilities = exact_binary_2x2_distribution()
        self.assertTrue(np.allclose([np.count_nonzero(np.isclose(gaps, s)) / 16 for s in expected_s], probabilities))
        self.assertAlmostEqual(float(np.dot(expected_s, probabilities)), 2 + math.sqrt(2))
        self.assertAlmostEqual(float(np.dot(normalized_s, probabilities)), 1)

    def test_normalization(self):
        normalized, original_mean = normalize_spacings(np.array([1, 2, 3, 4]))
        self.assertEqual(original_mean, 2.5)
        self.assertAlmostEqual(float(normalized.mean()), 1)

    def test_reject_nonpositive_mean(self):
        with self.assertRaises(ValueError):
            normalize_spacings(np.zeros(10))

    def test_wigner_density_zero_and_negative(self):
        self.assertAlmostEqual(float(wigner_density(0)), 0)
        self.assertAlmostEqual(float(wigner_density(-1)), 0)
        self.assertAlmostEqual(float(wigner_cdf(0)), 0)
        self.assertAlmostEqual(float(wigner_cdf(-1)), 0)

    def test_wigner_cdf(self):
        self.assertAlmostEqual(float(wigner_cdf(2)), 1 - math.exp(-math.pi))
        self.assertAlmostEqual(float(wigner_cdf(100)), 1)

    def test_wigner_integral_approximately_one(self):
        xs = np.linspace(0, 10, 3001)
        density = wigner_density(xs)
        integral = float(np.sum((density[:-1] + density[1:]) * np.diff(xs) / 2))
        self.assertAlmostEqual(integral, 1, delta=1e-5)

    def test_empirical_ks_bounds(self):
        self.assertAlmostEqual(ks_distance(np.array([0.0, 0.0, 0.0])), 1.0)
        self.assertGreaterEqual(ks_distance(np.array([0.2, 1.0, 2.0])), 0)
        self.assertLessEqual(ks_distance(np.array([0.2, 1.0, 2.0])), 1)

    def test_eigenvalue_gap_invalid_inputs(self):
        with self.assertRaises(ValueError):
            neighbor_spacings(np.ones((2, 3)), 0)
        with self.assertRaises(ValueError):
            neighbor_spacings(np.array([[[1., 2.], [0., 1.]]]), 0)
        with self.assertRaises(ValueError):
            neighbor_spacings(np.eye(2)[None], 1)

    def test_all_six_ensembles_normalized(self):
        rng = np.random.default_rng(9)
        for distribution in ("normal", "rademacher"):
            for size in (2, 4, 16):
                with self.subTest(distribution=distribution, size=size):
                    result = analyze_ensemble(size, 300, distribution, rng)
                    self.assertEqual(result.spacings.size, 300)
                    self.assertAlmostEqual(float(result.spacings.mean()), 1, delta=1e-12)
                    self.assertGreaterEqual(result.ks_distance, 0)
                    self.assertLessEqual(result.ks_distance, 1)

    def test_ks_is_order_independent(self):
        vals = np.array([1., 1., 0., 2., 1.])
        self.assertEqual(ks_distance(vals), ks_distance(vals[::-1]))


if __name__ == "__main__":
    unittest.main()
