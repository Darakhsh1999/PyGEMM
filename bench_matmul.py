import matrix
import matmul
import numpy as np
import matplotlib.pyplot as plt
from bench import bench

n_iterations = 10
N = [16,32,64,128,256,512]
N = np.arange(10,100)
T = np.zeros((5,len(N)))

for n_idx, n in enumerate(N):
    print(f"n = {n}")

    A = matrix.randint(n,n)
    B = matrix.randint(n,n)

    C = matrix.zeros(n,n)
    t1, _ = bench(matmul.matmul_row_brute, n_iterations, A, B, C)

    C = matrix.zeros(n,n)
    t2, _ = bench(matmul.matmul_col_brute, n_iterations, A, B, C)

    C = matrix.zeros(n,n)
    t3, _ = bench(matmul.matmul_row_factored, n_iterations, A, B, C)

    C = matrix.zeros(n,n)
    t4, _ = bench(matmul.matmul_col_factored, n_iterations, A, B, C)

    C = matrix.zeros(n,n)
    t5, _ = bench(matmul.matmul_factored, n_iterations, A, B, C)

    T[:,n_idx] = [t1,t2,t3,t4,t5]

plt.plot(N, T[0,:])
plt.plot(N, T[1,:])
plt.plot(N, T[2,:])
plt.plot(N, T[3,:])
plt.plot(N, T[4,:])
plt.legend(["Row brute", "Col brute", "Row factored", "Col factored", "Factored"])
plt.grid()
plt.show()