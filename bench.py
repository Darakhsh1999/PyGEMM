import time
import numpy as np
import matrix
import matmul

def bench(bench_name, fun, A, B, C, iterations, *args):
    """ benchmark function func by performing matrix multiplication
        between matrix A and B storing the value in C, iterations 
        number of times """

    t0 = time.time()
    for _ in range(iterations):
        fun(A, B, C, *args)
    t_tot = time.time() - t0
    print(bench_name + f" benchmark time is: {t_tot:.3f} s")
    return t_tot


if __name__ == '__main__':

    N = 512
    k = 64
    n_threads = 8
    n_iterations = 1
    A = matrix.rand(N,N)
    B = matrix.rand(N,N)
    A_np = np.array(A)
    B_np = np.array(B)
    C_np = A_np @ B_np

    
    C = matrix.zeros(N,N)
    bench("row brute", matmul.matmul_row_brute, A, B, C, n_iterations, N, N, N)
    C = matrix.zeros(N,N)
    bench("col brute", matmul.matmul_col_brute, A, B, C, n_iterations, N, N, N)
    C = matrix.zeros(N,N)
    bench("factored row", matmul.matmul_row_factored, A, B, C, n_iterations, N, N)
    C = matrix.zeros(N,N)
    bench(f"block = {k}", matmul.matmul_block, A, B, C, n_iterations, N, N, N, k)
    C = matrix.zeros(N,N)
    bench(f"thread = {n_threads}", matmul.matmul_thread, A, B, C, n_iterations, n_threads, N, N, N)