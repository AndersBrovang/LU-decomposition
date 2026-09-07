# LU Decomposition (with Partial Pivoting)

Factors a square matrix `A` into `L` and `U` such that (up to row order)
`A = L @ U`, using Gaussian elimination with partial pivoting.

## What it does

`lu_decompose(A)` returns two matrices:

- **`L`** — unit lower triangular (1s on the diagonal). Each entry below
  the diagonal is the multiplier used to eliminate that position during
  Gaussian elimination — `L` is effectively a record of the elimination
  steps, not a separate computation.
- **`U`** — upper triangular. This is the direct result of running
  elimination on `A` until everything below the diagonal is zero.

## Why partial pivoting

Back-substitution divides by the pivot value. If the pivot is small
relative to the other numbers involved, any existing rounding error
gets amplified by roughly `1 / pivot` — a pivot of `0.0001` can turn a
tiny error into a completely wrong answer. Partial pivoting avoids this
by always selecting the largest-magnitude entry available in each
column as the pivot, before eliminating with it. That's what the
`column_slice` / `argmax` / row-swap logic at the top of each loop
iteration is doing.

## How the function works, step by step

1. Set up `n` (matrix size), `L` as an identity matrix, and `U_work` as
   a float copy of `A` (elimination happens on this copy, not on `A`
   directly).
2. For each column `col` from left to right:
   - Search the column from row `col` downward for the largest-magnitude
     entry — that row becomes `pivot_row`.
   - If `pivot_row` differs from `col`, swap those two rows in both
     `U_work` and `L` (the swap in `L` only touches the columns before
     `col`, since those are the only ones with real values so far).
   - For every row below the pivot: compute the multiplier that zeroes
     out that row's entry in this column, store it in `L`, and subtract
     `multiplier * pivot_row` from the current row in `U_work`.
3. Once all columns are processed, `U_work` has become `U`.

## Usage

```python
import numpy as np
from lu_decomposition import lu_decompose

A = np.array([
    [2.0, 1.0, 1.0],
    [4.0, 3.0, 3.0],
    [8.0, 7.0, 9.0],
])

L, U = lu_decompose(A)
print(L)
print(U)
```

## Known limitation

This implementation does **not** track a permutation matrix `P`. That
means if pivoting caused any row swaps, `L @ U` will reconstruct a
row-swapped version of `A`, not `A` itself — so `np.allclose(L @ U, A)`
can fail even though the decomposition is correct. Tracking `P` (and
using it to permute `b` before solving `Ax = b`) would be the natural
next addition.
