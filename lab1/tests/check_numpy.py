"""Optional, reproducible comparison with NumPy (not used by the solver).

Run from the project root: python tests/check_numpy.py
Requires: pip install numpy
"""
from __future__ import annotations

from pathlib import Path
import random
import sys

try:
    import numpy as np
except ImportError:
    print("NumPy is needed only for the optional cross-check: python -m pip install numpy", file=sys.stderr)
    raise SystemExit(2)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jacobi_eigen import jacobi_eigendecomposition  # noqa: E402

SIZES = (2, 3, 4, 5, 8, 12, 16, 20)
COUNT_PER_SIZE = 30
SEED = 20261016
MAX_EIGENVALUE_ERROR_ALLOWED = 1e-10
MAX_RESIDUAL_ALLOWED = 1e-9
MAX_ORTHOGONALITY_ALLOWED = 1e-10


def main() -> int:
    rng = random.Random(SEED)
    max_eig = max_residual = max_orth = 0.0
    cases = 0
    for n in SIZES:
        for _ in range(COUNT_PER_SIZE):
            a = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i, n):
                    a[i][j] = a[j][i] = rng.uniform(-3.0, 3.0)
            answer = jacobi_eigendecomposition(a)
            aa = np.asarray(a)
            v = np.asarray(answer.eigenvectors)
            eig = np.asarray(answer.eigenvalues)
            max_eig = max(max_eig, float(np.max(np.abs(eig - np.linalg.eigvalsh(aa)))))
            max_residual = max(max_residual, float(np.linalg.norm(aa @ v - v * eig, ord='fro')))
            max_orth = max(max_orth, float(np.linalg.norm(v.T @ v - np.eye(n), ord='fro')))
            cases += 1
    success = (max_eig < MAX_EIGENVALUE_ERROR_ALLOWED
               and max_residual < MAX_RESIDUAL_ALLOWED
               and max_orth < MAX_ORTHOGONALITY_ALLOWED)
    print(f"Seed: {SEED}")
    print(f"Cases: {cases} ({COUNT_PER_SIZE} per size, sizes {SIZES})")
    print(f"Maximum absolute eigenvalue difference: {max_eig:.12e}")
    print(f"Maximum ||A V - V diag(lambda)||_F: {max_residual:.12e}")
    print(f"Maximum ||V^T V - I||_F: {max_orth:.12e}")
    print("Result:", "PASS" if success else "FAIL")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
