"""Public API of the Jacobi eigendecomposition package."""

from .core import (
    ConvergenceError,
    EigenResult,
    InvalidMatrixError,
    InvalidParameterError,
    JacobiError,
    NumericalError,
    jacobi_eigendecomposition,
)

__all__ = [
    "ConvergenceError",
    "EigenResult",
    "InvalidMatrixError",
    "InvalidParameterError",
    "JacobiError",
    "NumericalError",
    "jacobi_eigendecomposition",
]
