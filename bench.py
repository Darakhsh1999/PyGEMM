import time
import numpy as np
import matrix
import matmul

def bench(fun, A, B, C, iterations, *args):
    """ Matrix mul between object A and B"""

    t0 = time.time()
    for _ in range(iterations):
        C = fun(A, B, C, *args)
    t_tot = time.time() - t0
    return t_tot


if __name__ == '__main__':

    N = 32 
    A = matrix.randint(N,N)
    B = matrix.randint(N,N)
    A_np = np.array(matrix.randint(N,N))
    B_np = np.array(matrix.randint(N,N))
    C1 = matrix.zeros(N,N)
    C2 = matrix.zeros(N,N)
    C3 = matrix.zeros(N,N)
    C4 = matrix.zeros(N,N)

    C1 = matmul.matmul_row_brute(A, B, C1, N, N, N)
    C2 = matmul.matmul_col_brute(A, B, C2, N, N, N)
    C3 = matmul.matmul_row_factored(A, B, C3, N, N)
    C4 = matmul.matmul_block(A, B, C4, N, N, N, 4)

    t_row_brute = bench(matmul.matmul_row_brute, A, B, C1, 100, N, N, N)
    t_col_brute = bench(matmul.matmul_col_brute, A, B, C2, 100, N, N, N)
    t_row_factored = bench(matmul.matmul_row_factored, A, B, C3, 100, N, N)
    t_block = bench(matmul.matmul_block, A, B, C4, 100, N, N)

    print(t_row_brute)
    print(t_col_brute)
    print(t_row_factored)
