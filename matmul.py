""" Implementations of matrix multiplication A*B, shape A [n,q], shape B [q,m] """

import math
import matrix
import threading
import numpy as np
from matrix import LinearMatrix


def assert_algorithm(C, C_truth):
    """ Compares own implementation to numpy (ground truth) """
    C_np = np.array(C)
    assert C_np.shape == C_truth.shape, f"Invalid shape expected {C_truth.shape}, got {C_np.shape}"
    assert(np.less(np.abs(C_np-C_truth), 0.01).all())


def matmul_np(A, B):
    """ numpy internal matmul """
    return A @ B

def matmul_row_brute(A, B, C):
    """ Row major brute force """
    n, q, m = len(A), len(A[0]), len(B[0])
    for row in range(n):
        for col in range(m):
            for k in range(q):
                C[row][col] += A[row][k] * B[k][col] 

def matmul_col_brute(A, B, C):
    """ Column major brute force """
    n, q, m = len(A), len(A[0]), len(B[0])
    for col in range(m):
        for row in range(n):
            for k in range(q):
                C[row][col] += A[row][k] * B[k][col] 

def matmul_row_factored(A, B, C):
    """ Factored row vector in A """
    n, m = len(A), len(B[0])
    for row in range(n):
        a = A[row]
        for col in range(m):
            for k, a_val in enumerate(a):
                C[row][col] += a_val * B[k][col] 

def matmul_col_factored(A, B, C):
    """ Factored column vector in B """
    n, m = len(A), len(B[0])
    for col in range(m):
        b = matrix.get_col(B, col) 
        for row in range(n):
            for k, b_val in enumerate(b):
                C[row][col] += A[row][k] * b_val

def matmul_factored(A, B, C):
    """ Factored both row in A and column in B """
    n, q, m = len(A), len(A[0]), len(B[0])
    for row in range(n):
        a = A[row]
        for col in range(m):
            b = matrix.get_col(B, col)
            for k in range(q):
                C[row][col] += a[k] * b[k]

def matmul_block(A, B, C, block_size):
    """ Partition matrix into blocks """
    # A (n,q) : (p,s)
    # B (q,m) : (s,r)
    # p = n//k, s = q//k, r = m//k
    n, q, m = len(A), len(A[0]), len(B[0])
    k = block_size
    assert(n % k == 0)
    assert(q % k == 0)
    assert(m % k == 0)

    for p in range(0, n, k): # loop through A rows
        for r in range(0, m, k): # loop through B cols
            for s in range(0, q, k): # loop through block rows
                block_A = matrix.slice_2d(A, p, p+k, s, s+k) # (k,k) 
                block_B = matrix.slice_2d(B, s, s+k, r, r+k) # (k,k) 

                for row in range(k):
                    for col in range(k):
                        for l in range(k):
                            C[p+row][r+col] += block_A[row][l] * block_B[l][col] 

def matmul_row_brute_range(A, B, C, low, high, q, m):
    """ Row major brute force used for threaded matrix multiplication """
    for row in range(low, high):
        for col in range(m):
            for k in range(q):
                C[row][col] += A[row][k] * B[k][col] 

def matmul_thread(A, B, C, n_threads):
    """ Multithreaded matmul """
    n, q, m = len(A), len(A[0]), len(B[0])
    
    # Create threads
    s_thr_block = math.ceil(n/n_threads) 
    threads = []
    for thr_idx in range(n_threads):
        low = thr_idx*s_thr_block
        high = (thr_idx+1)*s_thr_block if (thr_idx+1)*s_thr_block <= n else n
        thread_i = threading.Thread(target=matmul_row_brute_range, args=(A, B, C, low, high, q, m))
        thread_i.start()
        threads.append(thread_i)
        
    # Join threads
    for thr_idx in range(n_threads): threads[thr_idx].join()
    
    return

def matmul_process(A, B, C, n_processes):
    pass

def matmul_linear_brute(A: LinearMatrix, B: LinearMatrix, C: LinearMatrix):
    """ LinearMatrix brute matmul """
    n, q, m = A.n_rows, A.n_cols, B.n_cols
    for row in range(n):
        for col in range(m):
            c_idx = row*q + col
            for k in range(q):
                C[c_idx] += A[row*q + k] * B[k*q + col] 