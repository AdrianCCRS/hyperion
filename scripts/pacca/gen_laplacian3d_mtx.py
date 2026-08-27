#!/usr/bin/env python3
"""Genera una matriz SPD dispersa sintética -- el Laplaciano 3D de 7
puntos sobre una malla N x N x N (discretización estándar por diferencias
finitas del operador de Poisson), en formato Matrix Market (coordinate,
real, symmetric, solo triángulo superior).

Por qué esta matriz y no una descargada: es SPD por construcción (la
factorización de Cholesky de CHOLMOD la exige), su tamaño es controlable
con un solo parámetro (N), y no depende de bajar nada de la SuiteSparse
Matrix Collection -- mismo criterio que el grafo Kronecker sintético de
GAP (--g 22): sintético declarado, no un archivo externo sin trazabilidad.

Uso: python3 gen_laplacian3d_mtx.py <N> <archivo_salida.mtx>
"""
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(f"uso: {sys.argv[0]} <N> <salida.mtx>", file=sys.stderr)
        return 1
    n = int(sys.argv[1])
    out_path = sys.argv[2]
    dim = n * n * n

    def idx(i: int, j: int, k: int) -> int:
        return i * n * n + j * n + k + 1  # Matrix Market es base-1

    entries = []  # (row, col, val), row <= col (triangulo superior)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                row = idx(i, j, k)
                entries.append((row, row, 6.0))
                for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    ni, nj, nk = i + di, j + dj, k + dk
                    if ni < n and nj < n and nk < n:
                        col = idx(ni, nj, nk)
                        r, c = (row, col) if row < col else (col, row)
                        entries.append((r, c, -1.0))

    with open(out_path, "w") as handle:
        handle.write("%%MatrixMarket matrix coordinate real symmetric\n")
        handle.write(f"{dim} {dim} {len(entries)}\n")
        for r, c, v in entries:
            handle.write(f"{r} {c} {v}\n")

    print(f"N={n} dim={dim} nnz_triangulo_superior={len(entries)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
