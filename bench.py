import time
from tqdm import tqdm

def bench(fun, n_iterations, *args):
    """ Benchmark function, returns timings: (t_tot,t_iter) """

    t0 = time.time()
    for _ in range(n_iterations):
        fun(*args)
    t_tot = time.time() - t0
    return t_tot, t_tot/n_iterations


#if __name__ == '__main__':

    #N = 512
    #k = 64
    #n_threads = 8
    #n_iterations = 1
    #A = matrix.rand(N,N)
    #B = matrix.rand(N,N)
    #A_np = np.array(A)
    #B_np = np.array(B)
    #C_np = A_np @ B_np

    
    ##C = matrix.zeros(N,N)
    ##bench("row brute", matmul.matmul_row_brute, A, B, C, n_iterations, N, N, N)
    ##C = matrix.zeros(N,N)
    ##bench("col brute", matmul.matmul_col_brute, A, B, C, n_iterations, N, N, N)
    ##C = matrix.zeros(N,N)
    ##bench("factored row", matmul.matmul_row_factored, A, B, C, n_iterations, N, N)
    ##C = matrix.zeros(N,N)
    ##bench(f"block = {k}", matmul.matmul_block, A, B, C, n_iterations, N, N, N, k)
    ##C = matrix.zeros(N,N)
    ##bench(f"thread = {n_threads}", matmul.matmul_thread, A, B, C, n_iterations, n_threads, N, N, N)