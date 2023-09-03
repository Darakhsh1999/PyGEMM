import matrix
import matmul
import numpy as np
import matplotlib.pyplot as plt
from bench import bench

if __name__ == "__main__":

    show_images = False

    ##### Bench matmul methods

    n_iterations = 15
    N = [16,32,64,128,256]
    T = np.zeros((13,len(N)))

    for n_idx, n in enumerate(N):
        print(f"n = {n}")

        A = matrix.randint(n,n)
        B = matrix.randint(n,n)
        A_lin = matrix.LinearMatrix((n,n), "randint")
        B_lin = matrix.LinearMatrix((n,n), "randint")

        # Regular matmuls
        t1, _ = bench(matmul.matmul_row_brute, n_iterations, A, B)
        t2, _ = bench(matmul.matmul_col_brute, n_iterations, A, B)
        t3, _ = bench(matmul.matmul_row_factored, n_iterations, A, B)
        t4, _ = bench(matmul.matmul_col_factored, n_iterations, A, B)
        t5, _ = bench(matmul.matmul_factored, n_iterations, A, B)
        t6, _ = bench(matmul.matmul_block, n_iterations, A, B, 8)
        t7, _ = bench(matmul.matmul_thread, n_iterations, A, B, 8)

        # LinearMatrix matmuls
        t8, _ = bench(matmul.matmul_linear_row_brute, n_iterations, A_lin, B_lin)
        t9, _ = bench(matmul.matmul_linear_col_brute, n_iterations, A_lin, B_lin)
        t10, _ = bench(matmul.matmul_linear_row_factored, n_iterations, A_lin, B_lin)
        t11, _ = bench(matmul.matmul_linear_col_factored, n_iterations, A_lin, B_lin)
        t12, _ = bench(matmul.matmul_linear_factored, n_iterations, A_lin, B_lin)
        t13, _ = bench(matmul.matmul_linear_block, n_iterations, A_lin, B_lin, 8)

        t = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13]
        T[:,n_idx] = t

    plt.figure(figsize=(12,8))
    for idx in range(len(t)):
        plt.plot(N, T[idx,:], "o-", linewidth=2)
    plt.legend([
        "Row brute",
        "Col brute",
        "Row factored",
        "Col factored",
        "Factored",
        "Block",
        "Thread",
        "Linear row brute",
        "Linear col brute",
        "Linear row factored",
        "Linear col factored",
        "Linear factored",
        "Linear block"
        ], loc="upper left")
    plt.grid()
    plt.title("Matmul benchmarks", fontsize=20)
    plt.xlabel("N", fontsize=16)
    plt.ylabel("$t_{tot} [s]$", fontsize=16)
    plt.xticks(N)
    plt.savefig("matmul_bench.png")
    if show_images:
        plt.show()

    ##### Bench multithreading
    n_iterations = 20
    N = [16,32,64,128]
    T = np.zeros((8,len(N)))

    for n_idx, n in enumerate(N):
        print(f"n = {n}")

        A = matrix.randint(n,n)
        B = matrix.randint(n,n)

        t1, _ = bench(matmul.matmul_thread, n_iterations, A, B, 1)
        t2, _ = bench(matmul.matmul_thread, n_iterations, A, B, 2)
        t3, _ = bench(matmul.matmul_thread, n_iterations, A, B, 3)
        t4, _ = bench(matmul.matmul_thread, n_iterations, A, B, 4)
        t5, _ = bench(matmul.matmul_thread, n_iterations, A, B, 6)
        t6, _ = bench(matmul.matmul_thread, n_iterations, A, B, 8)
        t7, _ = bench(matmul.matmul_thread, n_iterations, A, B, 12)
        t8, _ = bench(matmul.matmul_thread, n_iterations, A, B, 16)

        t = [t1,t2,t3,t4,t5,t6,t7,t8]
        T[:,n_idx] = t

    plt.figure(figsize=(12,8))
    for idx in range(len(t)):
        plt.plot(N, T[idx,:], "o-", linewidth=2)
    plt.legend([
        f"T = 1",
        f"T = 2",
        f"T = 3",
        f"T = 4",
        f"T = 6",
        f"T = 8",
        f"T = 12",
        f"T = 16"
        ], loc="upper left")
    plt.grid()
    plt.title("Multithreading benchmarks", fontsize=20)
    plt.xlabel("N", fontsize=16)
    plt.ylabel("$t_{tot}$ [s]", fontsize=16)
    plt.xticks(N)
    plt.savefig("threading_bench.png")
    if show_images:
        plt.show()
    
    ##### Bench multiprocessing
    n_iterations = 4
    N = [16,32,64,128]
    T = np.zeros((7,len(N)))

    for n_idx, n in enumerate(N):
        print(f"n = {n}")

        A = matrix.randint(n,n)
        B = matrix.randint(n,n)

        t1, _ = bench(matmul.matmul_process, n_iterations, A, B, 1)
        t2, _ = bench(matmul.matmul_process, n_iterations, A, B, 2)
        t3, _ = bench(matmul.matmul_process, n_iterations, A, B, 3)
        t4, _ = bench(matmul.matmul_process, n_iterations, A, B, 4)
        t5, _ = bench(matmul.matmul_process, n_iterations, A, B, 6)
        t6, _ = bench(matmul.matmul_process, n_iterations, A, B, 8)
        t7, _ = bench(matmul.matmul_process, n_iterations, A, B, 12)

        t = [t1,t2,t3,t4,t5,t6,t7]
        T[:,n_idx] = t

    plt.figure(figsize=(12,8))
    for idx in range(len(t)):
        plt.plot(N, T[idx,:], "o-", linewidth=2)
    plt.legend([
        f"P = 1",
        f"P = 2",
        f"P = 3",
        f"P = 4",
        f"P = 6",
        f"P = 8",
        f"P = 12"
        ], loc="upper left")

    plt.grid()
    plt.title("Multiprocessing benchmarks", fontsize=20)
    plt.xlabel("N", fontsize=16)
    plt.ylabel("$t_{tot}$ [s]", fontsize=16)
    plt.xticks(N)
    plt.savefig("multiprocess_bench.png")
    if show_images:
        plt.show()