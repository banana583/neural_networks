"""Запуск эксперимента: python -m random_matrix_lab."""

import argparse
import csv
from pathlib import Path
import numpy as np
from .analysis import SIZES, analyze_ensemble, plot_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Лабораторная №2: интервалы собственных чисел случайных матриц")
    parser.add_argument("--count", type=int, default=10000, help="Матриц в каждом ансамбле (по умолчанию: 10000)")
    parser.add_argument("--seed", type=int, default=20261017, help="Начальное значение генератора случайных чисел")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Каталог выходных данных")
    args = parser.parse_args(argv)
    if args.count < 1000:
        parser.error("По заданию в каждом ансамбле должно быть не менее 1000 матриц")

    args.output.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output / "plots"
    rng = np.random.default_rng(args.seed)
    rows = []
    print(f"Число матриц на ансамбль: {args.count}; seed: {args.seed}")
    for distribution in ("normal", "rademacher"):
        for size in SIZES:
            result = analyze_ensemble(size, args.count, distribution, rng)
            np.savetxt(args.output / f"spacings_{distribution}_{size}.csv", result.spacings,
                       delimiter=",", header="normalized_spacing", comments="")
            plot_result(result, plot_dir / f"{distribution}_{size}.png")
            rows.append({
                "distribution": distribution,
                "size": size,
                "matrices_count": args.count,
                "left_eigenvalue_index_1based": result.gap_left_index + 1,
                "right_eigenvalue_index_1based": result.gap_left_index + 2,
                "raw_mean_gap": f"{result.original_mean_gap:.12g}",
                "normalized_mean": f"{float(np.mean(result.spacings)):.12g}",
                "zero_fraction": f"{result.zero_fraction:.12g}",
                "ks_to_wigner": f"{result.ks_distance:.12g}",
            })
            print(f"{distribution:11} n={size:2d}: mean={result.original_mean_gap:.6f}; "
                  f"normalized={np.mean(result.spacings):.6f}; "
                  f"zeros={result.zero_fraction:.2%}; D={result.ks_distance:.4f}")

    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Готово: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
