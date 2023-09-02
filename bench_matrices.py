from matrix import *
from bench import bench
from functools import partial


n_iterations = 2000
N = [10, 100, 200, 500, 1000]

T1 = []
for n in N:
    fun = partial(zeros, n_rows=n, n_cols=n)
    t_tot, t_iter = bench(fun, n_iterations)
    T1.append((t_tot, t_iter))

T2 = []
for n in N:
    fun = partial(ones, n_rows=n, n_cols=n)
    t_tot, t_iter = bench(fun, n_iterations)
    T2.append((t_tot, t_iter))

T3 = []
for n in N:
    fun = partial(fill, n_rows=n, n_cols=n, q=1024)
    t_tot, t_iter = bench(fun, n_iterations)
    T3.append((t_tot, t_iter))

print("Printing benchmarks...")
for idx, n in enumerate(N):
    print(f"N = {n}, fun1 = {T1[idx][1]:.4f}, fun2= {T2[idx][1]:.4f}, fun3 = {T3[idx][1]:.4f}")