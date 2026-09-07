import numpy as np

def lu_decompose(A):
    n = A.shape[0]
    L = np.eye(n)
    U_work = A.astype(float).copy()

    for col in range(n - 1):
        column_slice = U_work[col:, col]
        abs_slice = np.abs(column_slice)
        local_index = np.argmax(abs_slice)
        pivot_row = local_index + col

        if pivot_row != col:
            U_work[[col, pivot_row]] = U_work[[pivot_row, col]]
            L[[col, pivot_row], :col] = L[[pivot_row, col], :col]

        pivot = U_work[col, col]

        for row in range(col + 1, n):
            multiplier = U_work[row, col] / pivot
            L[row, col] = multiplier
            U_work[row, col:] -= multiplier * U_work[col, col:]

    U = U_work
    return L, U