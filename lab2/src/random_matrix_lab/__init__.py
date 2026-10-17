"""Практическая часть лабораторной № 2 «Свойства случайных матриц»."""

from .analysis import (
    EnsembleResult, analyze_ensemble, central_gap_index, exact_binary_2x2_distribution,
    generate_ensemble, ks_distance, neighbor_spacings, normalize_spacings,
    wigner_cdf, wigner_density,
)

__all__ = [
    "EnsembleResult", "analyze_ensemble", "central_gap_index",
    "exact_binary_2x2_distribution", "generate_ensemble", "ks_distance",
    "neighbor_spacings", "normalize_spacings", "wigner_cdf", "wigner_density",
]
