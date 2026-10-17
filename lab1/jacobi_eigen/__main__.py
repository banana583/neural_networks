"""Command-line interface: python -m jacobi_eigen matrix.json."""

from __future__ import annotations

import argparse
import json
import sys

from . import JacobiError, jacobi_eigendecomposition


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jacobi-eigen",
        description="Eigenvalues and eigenvectors of a finite real symmetric matrix",
    )
    parser.add_argument("file", nargs="?", default="-", help="input JSON file ('-' for stdin)")
    parser.add_argument("--epsilon", type=float, default=1e-24, help="upper-triangle sum-of-squares bound")
    parser.add_argument("--max-sweeps", type=int, default=100, help="maximum number of full cyclic passes")
    parser.add_argument("--no-sort", action="store_true", help="keep diagonal order instead of sorting")
    args = parser.parse_args(argv)
    try:
        if args.file == "-":
            data = json.load(sys.stdin)
        else:
            with open(args.file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        # Accept either a raw matrix [[...], [...]] or {"matrix": [[...]]}.
        matrix = data["matrix"] if isinstance(data, dict) else data
        result = jacobi_eigendecomposition(
            matrix,
            epsilon=args.epsilon,
            max_sweeps=args.max_sweeps,
            sort=not args.no_sort,
        )
    except (JacobiError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    json.dump(
        {
            "eigenvalues": result.eigenvalues,
            "eigenvectors": result.eigenvectors,
            "sweeps": result.sweeps,
            "rotations": result.rotations,
            "off_diagonal_norm": result.off_diagonal_norm,
        },
        sys.stdout,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
