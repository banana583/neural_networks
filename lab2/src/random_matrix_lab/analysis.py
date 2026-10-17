"""Исследование расстояний между соседними уровнями случайных симметричных матриц."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Построение графиков работает без графического интерфейса.
import matplotlib.pyplot as plt

Distribution = Literal["normal", "rademacher"]
SIZES = (2, 4, 16)


@dataclass(frozen=True)
class EnsembleResult:
    """Результаты одного численного ансамбля."""

    distribution: Distribution
    size: int
    matrices_count: int
    gap_left_index: int  # Индексация с нуля.
    original_mean_gap: float
    spacings: np.ndarray
    zero_fraction: float
    ks_distance: float


def central_gap_index(size: int) -> int:
    """Индекс первого уровня центральной пары при сортировке по возрастанию."""
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 2:
        raise ValueError("Размер матрицы должен быть целым числом не меньше 2")
    return (size - 2) // 2


def generate_ensemble(
    size: int,
    count: int,
    distribution: Distribution,
    rng: np.random.Generator,
) -> np.ndarray:
    """Создать A=X+X.T для независимых X_ij ~ N(0,1) либо P(X_ij=+-1)=1/2.

    В результате нормальные диагонали имеют σ=2, а внедиагональные σ=sqrt(2).
    Для ±1 диагонали равны ±2, внедиагональные: −2, 0 либо 2.
    """
    central_gap_index(size)
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("Число матриц должно быть положительным целым")
    if distribution not in ("normal", "rademacher"):
        raise ValueError("Распределение должно быть normal или rademacher")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng должен быть numpy.random.Generator")

    if distribution == "normal":
        source = rng.standard_normal((count, size, size))
    else:
        source = 2.0 * rng.integers(0, 2, size=(count, size, size)) - 1.0
    return source + source.swapaxes(-2, -1)


def neighbor_spacings(matrices: np.ndarray, gap_left_index: int) -> np.ndarray:
    """Расстояния λ_(k+1)−λ_k, где k фиксирован для всего ансамбля."""
    m = np.asarray(matrices)
    if m.ndim != 3 or m.shape[-2] != m.shape[-1] or m.shape[-1] < 2:
        raise ValueError("Ожидается массив матриц формы (count, size, size), size >= 2")
    if not np.all(np.isfinite(m)) or not np.allclose(m, m.swapaxes(-1, -2), rtol=0, atol=1e-12):
        raise ValueError("Матрицы должны быть конечными и симметричными")
    if isinstance(gap_left_index, bool) or not isinstance(gap_left_index, (int, np.integer)) or not (0 <= gap_left_index < m.shape[-1] - 1):
        raise ValueError("Неверный индекс соседней пары")
    eigenvalues = np.linalg.eigvalsh(m)  # Для симметричных матриц собственные числа уже отсортированы.
    gaps = eigenvalues[:, gap_left_index + 1] - eigenvalues[:, gap_left_index]
    # Погрешность вычисления повторных собственных чисел может давать ~1e-14 вместо нуля.
    return np.maximum(gaps, 0.0)


def normalize_spacings(gaps: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(gaps, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Расстояния должны быть непустым массивом конечных неотрицательных чисел")
    mean_gap = float(np.mean(values))
    if mean_gap <= 0:
        raise ValueError("Среднее расстояние равно нулю: нормировка невозможна")
    return values / mean_gap, mean_gap


def wigner_density(x: np.ndarray | float) -> np.ndarray:
    """Догадка Вигнера для GOE с единичным средним расстоянием."""
    val = np.asarray(x, dtype=float)
    positive = np.maximum(val, 0.0)
    return np.where(val >= 0, (np.pi / 2) * positive * np.exp(-np.pi * positive**2 / 4), 0.0)


def wigner_cdf(x: np.ndarray | float) -> np.ndarray:
    val = np.asarray(x, dtype=float)
    positive = np.maximum(val, 0.0)
    return np.where(val >= 0, -np.expm1(-np.pi * positive**2 / 4), 0.0)


def ks_distance(spacings: np.ndarray) -> float:
    """Максимальное отличие эмпирической CDF от CDF Вигнера (описательная оценка)."""
    x = np.sort(np.asarray(spacings, dtype=float))
    if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("Ожидается непустой массив неотрицательных расстояний")
    n = x.size
    cdf = wigner_cdf(x)
    return float(np.max(np.maximum(np.arange(1, n + 1) / n - cdf, cdf - np.arange(n) / n)))


def analyze_ensemble(
    size: int, count: int, distribution: Distribution, rng: np.random.Generator,
) -> EnsembleResult:
    k = central_gap_index(size)
    matrices = generate_ensemble(size, count, distribution, rng)
    gaps = neighbor_spacings(matrices, k)
    normalized, mean_gap = normalize_spacings(gaps)
    return EnsembleResult(
        distribution=distribution,
        size=size,
        matrices_count=count,
        gap_left_index=k,
        original_mean_gap=mean_gap,
        spacings=normalized,
        zero_fraction=float(np.count_nonzero(normalized < 1e-10) / count),
        ks_distance=ks_distance(normalized),
    )


def plot_result(result: EnsembleResult, filepath: Path) -> None:
    """На левой оси гистограмма и функция Вигнера, на правой - их CDF."""
    fig, (ax, ax_cdf) = plt.subplots(1, 2, figsize=(11, 4.15), layout="constrained")
    x_max = max(3.6, min(5.0, float(np.quantile(result.spacings, 0.999)) * 1.08))
    edges = np.linspace(0, x_max, 42 if result.distribution == "normal" else 26)
    x = np.linspace(0, x_max, 600)
    ax.hist(result.spacings, bins=edges, density=True, color="#7197bc", alpha=0.78,
            edgecolor="white", linewidth=0.65, label="Численный ансамбль")
    ax.plot(x, wigner_density(x), color="#bd423a", lw=2.2, label="Формула Вигнера")
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Нормированный интервал s")
    ax.set_ylabel("Плотность / гистограмма")
    ax.set_title("Гистограмма и теоретическая кривая", fontsize=10)
    ax.grid(alpha=0.15, axis="y")
    ax.legend(loc="upper right", fontsize=8)

    sorted_gaps = np.sort(result.spacings)
    # «post»: скачок CDF в точке, где находится наблюдаемое значение.
    ax_cdf.step(np.r_[0.0, sorted_gaps, x_max], np.r_[0.0, np.arange(1, len(sorted_gaps)+1)/len(sorted_gaps), 1.0],
                where="post", color="#2d6b73", lw=1.7, label="Эмпирическая F(s)")
    ax_cdf.plot(x, wigner_cdf(x), color="#bd423a", lw=2, label="Теоретическая F(s)")
    ax_cdf.set_xlim(0, x_max)
    ax_cdf.set_ylim(0, 1.025)
    ax_cdf.set_xlabel("Нормированный интервал s")
    ax_cdf.set_ylabel("Накопленная вероятность")
    ax_cdf.set_title(f"Сравнение функций распределения (D = {result.ks_distance:.3f})", fontsize=10)
    ax_cdf.grid(alpha=0.17)
    ax_cdf.legend(loc="lower right", fontsize=8)

    distribution_name = "Нормальное" if result.distribution == "normal" else "Двухточечное (±1)"
    fig.suptitle(f"{distribution_name} распределение · размер {result.size}×{result.size} · {result.matrices_count} матриц", fontsize=12)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=190, bbox_inches="tight")
    plt.close(fig)


def exact_binary_2x2_distribution() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Три точных значения нормированного интервала и их вероятности."""
    s = np.array([0.0, 4.0, 4 * np.sqrt(2)])
    p = np.array([0.25, 0.50, 0.25])
    return s, s / np.dot(s, p), p
