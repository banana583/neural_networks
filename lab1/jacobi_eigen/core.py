"""Cyclic Jacobi eigensolver for finite real symmetric matrices.

Only the Python standard library is required. Eigenvectors are returned as
COLUMNS of a row-major matrix. See README.md for the public API contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Sequence


class JacobiError(Exception):
    """Base class for anticipated solver errors."""


class InvalidMatrixError(JacobiError, ValueError):
    """A matrix violates the solver's input contract."""


class InvalidParameterError(JacobiError, ValueError):
    """An algorithm setting violates the public API contract."""


class ConvergenceError(JacobiError, ArithmeticError):
    """The solver cannot meet the accuracy requirement in the allowed sweeps."""


class NumericalError(JacobiError, ArithmeticError):
    """The finite-precision input or output cannot be represented safely."""


@dataclass(frozen=True)
class EigenResult:
    """Eigenvalues and the corresponding eigenvectors (stored by columns).

    Attributes:
        eigenvalues: Eigenvalues in ascending order by default.
        eigenvectors: Rows of the eigenvector matrix V, so V[r][j] is the
            r-th coordinate of the eigenvector for eigenvalues[j].
        sweeps: Number of full cyclic sweeps performed.
        rotations: Number of nonzero Jacobi rotations performed.
        off_diagonal_norm: sqrt(sum(B[p][q]**2 for p < q)), in input units,
            where B = V.T @ A @ V to working precision.
    """

    eigenvalues: tuple[float, ...]
    eigenvectors: tuple[tuple[float, ...], ...]
    sweeps: int
    rotations: int
    off_diagonal_norm: float


def _validate_matrix(matrix: Sequence[Sequence[Real]]) -> list[list[float]]:
    if isinstance(matrix, (str, bytes)):
        raise InvalidMatrixError("matrix must be a non-empty square sequence")
    try:
        n = len(matrix)
    except (TypeError, ValueError) as exc:
        raise InvalidMatrixError("matrix must be a non-empty square sequence") from exc
    if n < 1:
        raise InvalidMatrixError("matrix size must be at least 1 x 1")
    a: list[list[float]] = []
    for i, row in enumerate(matrix):
        if isinstance(row, (str, bytes)):
            raise InvalidMatrixError(f"row {i} must be a sequence of numbers")
        try:
            if len(row) != n:
                raise InvalidMatrixError(f"row {i} has length {len(row)}, expected {n}")
        except (TypeError, ValueError) as exc:
            if isinstance(exc, InvalidMatrixError):
                raise
            raise InvalidMatrixError(f"row {i} must be a sequence") from exc
        converted: list[float] = []
        for j, element in enumerate(row):
            if isinstance(element, bool) or not isinstance(element, Real):
                raise InvalidMatrixError(f"matrix[{i}][{j}] must be a real number")
            try:
                value = float(element)
            except (OverflowError, TypeError, ValueError) as exc:
                raise InvalidMatrixError(f"matrix[{i}][{j}] cannot be stored as a float") from exc
            if not math.isfinite(value):
                raise InvalidMatrixError(f"matrix[{i}][{j}] must be finite")
            converted.append(value)
        a.append(converted)
    for i in range(n):
        for j in range(i + 1, n):
            if a[i][j] != a[j][i]:
                raise InvalidMatrixError(
                    f"matrix must be symmetric: [{i}][{j}] != [{j}][{i}]"
                )
    return a


def _validate_settings(epsilon: float, max_sweeps: int, sort: bool) -> tuple[float, int]:
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise InvalidParameterError("epsilon must be a positive finite real number")
    try:
        eps = float(epsilon)
    except (ValueError, OverflowError) as exc:
        raise InvalidParameterError("epsilon must be a positive finite real number") from exc
    if not math.isfinite(eps) or eps <= 0:
        raise InvalidParameterError("epsilon must be a positive finite real number")
    if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, Integral) or max_sweeps < 1:
        raise InvalidParameterError("max_sweeps must be a positive integer")
    if not isinstance(sort, bool):
        raise InvalidParameterError("sort must be a bool")
    return eps, int(max_sweeps)


def _off_diagonal_norm(a: list[list[float]]) -> float:
    """Upper-triangle Frobenius norm calculated without squaring/overflow."""
    result = 0.0
    for p in range(len(a)):
        for q in range(p + 1, len(a)):
            result = math.hypot(result, a[p][q])
    return result


def jacobi_eigendecomposition(
    matrix: Sequence[Sequence[Real]],
    *,
    epsilon: float = 1e-24,
    max_sweeps: int = 100,
    sort: bool = True,
) -> EigenResult:
    """Return the eigendecomposition of a finite real symmetric matrix.

    Preconditions:
      - `matrix` is square, nonempty, exactly symmetric, with finite real entries.
      - `epsilon` is finite and strictly positive.
      - `max_sweeps` is an integer >= 1; `sort` is a bool.

    Postconditions on successful return:
      - eigenvalues[j] and eigenvectors column j describe the same eigenpair.
      - eigenvectors are approximately orthonormal and A@V approximately V@D.
      - the sum of squares of upper-triangular off-diagonal elements of the
        computed diagonalized matrix is strictly below `epsilon` (subject to
        the limits of IEEE 754 floating point arithmetic).
      - the input matrix is not mutated.

    Raises:
      InvalidMatrixError: unsupported shape/values or nonsymmetric input.
      InvalidParameterError: invalid accuracy or iteration settings.
      ConvergenceError: limit reached or a floating-point rotation stalls.
      NumericalError: result is unrepresentable, or scaling loses input data.

    Note: epsilon is an *absolute squared* upper-triangle threshold, not a
    relative tolerance. Scaling by max(abs(A_ij)) reduces overflow risk.
    """
    eps, sweep_limit = _validate_settings(epsilon, max_sweeps, sort)
    original = _validate_matrix(matrix)
    n = len(original)

    # An already diagonal matrix can be solved exactly as stored. In particular,
    # do not scale [1e308, 1e-20] and inadvertently lose the smaller eigenvalue.
    if all(original[p][q] == 0.0 for p in range(n) for q in range(p + 1, n)):
        indices = sorted(range(n), key=lambda j: original[j][j]) if sort else list(range(n))
        return EigenResult(
            eigenvalues=tuple(original[j][j] for j in indices),
            eigenvectors=tuple(tuple(float(r == j) for j in indices) for r in range(n)),
            sweeps=0,
            rotations=0,
            off_diagonal_norm=0.0,
        )

    scale = max(abs(x) for row in original for x in row)
    a = [[x / scale for x in row] for row in original]
    # IEEE 754 can round a nonzero entry to 0 when the dynamic range is
    # extreme. Declaring convergence after discarding it would be incorrect
    # (for example, diag +/-1e308 with off-diagonal 1e-20 and eps=1e-100).
    # A controlled error is preferable to a silently incorrect eigenpair.
    if any(original[i][j] != 0.0 and a[i][j] == 0.0
           for i in range(n) for j in range(n)):
        raise NumericalError(
            "scaling would discard a nonzero matrix entry; "
            "the matrix dynamic range exceeds float precision"
        )
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    rotations = 0
    sweeps = 0
    # ||offdiag(A_scaled)||_2 * scale < sqrt(epsilon).
    threshold = math.sqrt(eps) / scale if scale else math.inf
    norm = _off_diagonal_norm(a)

    def done(offdiag: float) -> bool:
        # The separate zero branch handles a threshold that underflowed to 0.
        return offdiag == 0.0 or offdiag < threshold

    while not done(norm) and sweeps < sweep_limit:
        changed = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if apq == 0.0:
                    continue
                # t = tan(phi), |t| <= 1. This formulation avoids a division
                # by 2*apq, which can overflow for tiny apq.
                delta = (a[q][q] - a[p][p]) / 2.0
                if delta == 0.0:
                    t = 1.0
                else:
                    t = apq / (delta + math.copysign(math.hypot(delta, apq), delta))
                if t == 0.0:
                    # Loss of significance prevents further resolution.
                    continue
                c = 1.0 / math.hypot(1.0, t)
                s = t * c
                old_pp, old_qq = a[p][p], a[q][q]
                a[p][p] = old_pp - t * apq
                a[q][q] = old_qq + t * apq
                a[p][q] = 0.0
                a[q][p] = 0.0
                for r in range(n):
                    if r != p and r != q:
                        arp, arq = a[r][p], a[r][q]
                        new_rp = c * arp - s * arq
                        new_rq = s * arp + c * arq
                        a[r][p] = a[p][r] = new_rp
                        a[r][q] = a[q][r] = new_rq
                for r in range(n):
                    vrp, vrq = v[r][p], v[r][q]
                    v[r][p] = c * vrp - s * vrq
                    v[r][q] = s * vrp + c * vrq
                rotations += 1
                changed = True
        sweeps += 1
        norm = _off_diagonal_norm(a)
        if not changed and not done(norm):
            raise ConvergenceError(
                "Jacobi rotations cannot make further progress at this floating-point precision"
            )

    if not done(norm):
        raise ConvergenceError(
            f"accuracy epsilon={eps:g} not achieved after {sweep_limit} sweeps; "
            "increase max_sweeps or choose a numerically realistic epsilon"
        )
    indices = sorted(range(n), key=lambda j: a[j][j]) if sort else list(range(n))
    eigenvalues: list[float] = []
    for j in indices:
        value = a[j][j] * scale
        if not math.isfinite(value):
            raise NumericalError("an eigenvalue exceeds the finite float range")
        eigenvalues.append(value)
    eigenvectors = tuple(tuple(v[r][j] for j in indices) for r in range(n))
    residual_norm = norm * scale
    if not math.isfinite(residual_norm):
        raise NumericalError("off-diagonal residual exceeds the finite float range")
    return EigenResult(
        eigenvalues=tuple(eigenvalues),
        eigenvectors=eigenvectors,
        sweeps=sweeps,
        rotations=rotations,
        off_diagonal_norm=residual_norm,
    )
