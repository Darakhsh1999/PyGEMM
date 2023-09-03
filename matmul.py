""" Implementations of matrix multiplication A*B, shape A [n,q], shape B [q,m] """

import math
import matrix
import threading
import multiprocessing
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

def matmul(A,B):
    """ Matrix multiplication C = A*B """
    N, Q, M = len(A), len(A[0]), len(B[0])
    assert N == len(B), "Undefined matrix shapes"
    C = matrix.zeros(N,M)
    # TODO once we fight the best bench for different sizes we can create a general matmul function
    pass

def matmul_row_brute(A, B):
    """ Row major brute force """
    N, Q, M = len(A), len(A[0]), len(B[0])
    assert Q == len(B), "Undefined matrix shapes"
    C = matrix.zeros(N,M)
    for row in range(N):
        for col in range(M):
            for k in range(Q):
                C[row][col] += A[row][k] * B[k][col] 
    return C

def matmul_col_brute(A, B):
    """ Column major brute force """
    N, Q, M = len(A), len(A[0]), len(B[0])
    assert Q == len(B), "Undefined matrix shapes"
    C = matrix.zeros(N,M)
    for col in range(M):
        for row in range(N):
            for k in range(Q):
                C[row][col] += A[row][k] * B[k][col] 
    return C

def matmul_row_factored(A, B):
    """ Factored row vector in A """
    N, M = len(A), len(B[0])
    assert len(A[0]) == len(B), "Undefined matrix shapes"
    C = matrix.zeros(N,M)
    for row in range(N):
        a = A[row]
        for col in range(M):
            for k, a_val in enumerate(a):
                C[row][col] += a_val * B[k][col] 
    return C

def matmul_col_factored(A, B):
    """ Factored column vector in B """
    N, M = len(A), len(B[0])
    assert len(A[0]) == len(B), "Undefined matrix shapes"
    C = matrix.zeros(N,M)
    for col in range(M):
        b = matrix.get_col(B, col) 
        for row in range(N):
            for k, b_val in enumerate(b):
                C[row][col] += A[row][k] * b_val
    return C

def matmul_factored(A, B):
    """ Factored both row in A and column in B """
    N, Q, M = len(A), len(A[0]), len(B[0])
    assert Q == len(B), "Undefined matrix shapes"
    C = matrix.zeros(N,M)
    for row in range(N):
        a = A[row]
        for col in range(M):
            b = matrix.get_col(B, col)
            for k in range(Q):
                C[row][col] += a[k] * b[k]
    return C

def matmul_block(A, B, block_size):
    """ Partition matrix into blocks """
    # A (n,q) : (p,s)
    # B (q,m) : (s,r)
    # p = n//k, s = q//k, r = m//k
    N, Q, M = len(A), len(A[0]), len(B[0])
    assert Q == len(B), "Undefined matrix shapes"
    C = matrix.zeros(N,M)
    k = block_size
    assert(N % k == 0)
    assert(Q % k == 0)
    assert(M % k == 0)

    for p in range(0, N, k): # loop through A rows
        for r in range(0, M, k): # loop through B cols
            for s in range(0, Q, k): # loop through block rows
                block_A = matrix.slice_2d(A, p, p+k, s, s+k) # (k,k) 
                block_B = matrix.slice_2d(B, s, s+k, r, r+k) # (k,k) 

                for row in range(k):
                    for col in range(k):
                        for l in range(k):
                            C[p+row][r+col] += block_A[row][l] * block_B[l][col] 
    return C

def matmul_row_range(A, B, C, low, high, q, m):
    """ Row major brute force used for threaded matrix multiplication """
    for row in range(low, high):
        for col in range(m):
            for k in range(q):
                C[row][col] += A[row][k] * B[k][col] 

def matmul_thread(A, B, n_threads):
    """ Multithreaded matmul """
    N, Q, M = len(A), len(A[0]), len(B[0])
    assert Q == len(B), "Undefined matrix shapes"
    C = matrix.zeros(N,M)
    
    # Create threads
    s_block = math.ceil(N/n_threads) 
    threads = []
    for thr_idx in range(n_threads):
        low = thr_idx*s_block
        high = (thr_idx+1)*s_block if (thr_idx+1)*s_block <= N else N
        thread_i = threading.Thread(target=matmul_row_range, args=(A, B, C, low, high, Q, M))
        thread_i.start()
        threads.append(thread_i)
        
    # Join threads
    for thr_idx in range(n_threads): threads[thr_idx].join()
    return C

def matmul_row_range_process(A, B, C, low, high, q, m):
    """ Row major brute force used for multiprocess matrix multiplication """
    for row in range(low, high):
        for col in range(m):
            for k in range(q):
                C[row*m + col] += A[row][k] * B[k][col] 

def matmul_process(A, B, n_processes):
    N, Q, M = len(A), len(A[0]), len(B[0])
    assert Q == len(B), "Undefined matrix shapes"
    C = multiprocessing.Array("i", N*M) if isinstance(A[0][0], int) else multiprocessing.Array("d", N*M)

    # Create processes
    s_block = math.ceil(N/n_processes) 
    processes = []
    for p_idx in range(n_processes):
        low = p_idx*s_block
        high = (p_idx+1)*s_block if (p_idx+1)*s_block <= N else N
        p_i = multiprocessing.Process(target=matmul_row_range_process, args=(A, B, C, low, high, Q, M))
        p_i.start()
        processes.append(p_i)
        
    for p_idx in range(n_processes): processes[p_idx].join() # Join processes

    C = matrix.reshape_2d(C[:], n_rows=N, n_cols=M)
    return C


##### LinearMatrix matmuls

def matmul_linear_row_brute(A: LinearMatrix, B: LinearMatrix):
    """ LinearMatrix brute matmul """
    N, Q, M = A.n_rows, A.n_cols, B.n_cols
    assert Q == B.n_rows, "Undefined matrix shapes"
    C = LinearMatrix((N,M), "zeros")
    for row in range(N):
        for col in range(M):
            c_idx = row*M + col
            for k in range(Q):
                C[c_idx] += A[row*Q + k] * B[k*M + col] 
    return C

def matmul_linear_col_brute(A: LinearMatrix, B: LinearMatrix):
    """ LinearMatrix brute matmul """
    N, Q, M = A.n_rows, A.n_cols, B.n_cols
    assert Q == B.n_rows, "Undefined matrix shapes"
    C = LinearMatrix((N,M), "zeros")
    for col in range(M):
        for row in range(N):
            c_idx = row*M + col
            for k in range(Q):
                C[c_idx] += A[row*Q + k] * B[k*M + col] 
    return C

def matmul_linear_row_factored(A: LinearMatrix, B: LinearMatrix):
    N, Q, M = A.n_rows, A.n_cols, B.n_cols
    assert Q == B.n_rows, "Undefined matrix shapes"
    C = LinearMatrix((N,M), "zeros")
    for row in range(N):
        a = A[row*Q:(row+1)*Q]
        for col in range(M):
            for k, a_val in enumerate(a):
                C[row*M + col] += a_val * B[k*M + col] 
    return C

def matmul_linear_col_factored(A: LinearMatrix, B: LinearMatrix):
    N, Q, M = A.n_rows, A.n_cols, B.n_cols
    assert Q == B.n_rows, "Undefined matrix shapes"
    C = LinearMatrix((N,M), "zeros")
    for col in range(M):
        b = B[slice(col,(Q-1)*M+col+1,M)]
        for row in range(N):
            for k, b_val in enumerate(b):
                C[row*M + col] += A[row*Q + k] * b_val
    return C

def matmul_linear_factored(A: LinearMatrix, B: LinearMatrix):
    N, Q, M = A.n_rows, A.n_cols, B.n_cols
    assert Q == B.n_rows, "Undefined matrix shapes"
    C = LinearMatrix((N,M), "zeros")
    for row in range(N):
        a = A[row*Q:(row+1)*Q]
        for col in range(M):
            b = B[slice(col,(Q-1)*M+col+1,M)]
            for k in range(Q):
                C[row*M + col] += a[k] * b[k]
    return C

def matmul_linear_block(A: LinearMatrix, B: LinearMatrix, block_size: int):
    """ Partition matrix into blocks """
    # A (N,Q) : (p,s)
    # B (Q,M) : (s,r)
    # p = N//k, s = Q//k, r = M//k
    N, Q, M = A.n_rows, A.n_cols, B.n_cols
    assert Q == B.n_rows, "Undefined matrix shapes"
    C = matrix.LinearMatrix((N,M), "zeros")
    k = block_size
    assert(N % k == 0)
    assert(Q % k == 0)
    assert(M % k == 0)

    for p in range(0, N, k): # loop through A rows
        for r in range(0, M, k): # loop through B cols
            for s in range(0, Q, k): # loop through block rows
                block_A = A.slice_2d(p, s, k)
                block_B = B.slice_2d(s, r, k)

                for row in range(k):
                    for col in range(k):
                        for l in range(k):
                            C[(p+row)*M + (r+col)] += block_A[row*k + l] * block_B[l*k + col] 
    return C

def matmul_row_range_linear_process(A: LinearMatrix, B: LinearMatrix, C, low, high, q, m):
    for row in range(low, high):
        for col in range(m):
            for k in range(q):
                C[row*m + col] += A[row*q + k] * B[k*m + col] 

def matmul_linear_process(A, B, n_processes):
    N, Q, M = A.n_rows, A.n_cols, B.n_cols
    assert Q == B.n_rows, "Undefined matrix shapes"
    C = multiprocessing.Array("i", N*M) if isinstance(A[0], int) else multiprocessing.Array("d", N*M)

    # Create processes
    s_block = math.ceil(N/n_processes) 
    processes = []
    for p_idx in range(n_processes):
        low = p_idx*s_block
        high = (p_idx+1)*s_block if (p_idx+1)*s_block <= N else N
        p_i = multiprocessing.Process(target=matmul_row_range_linear_process, args=(A, B, C, low, high, Q, M))
        p_i.start()
        processes.append(p_i)
        
    for p_idx in range(n_processes): processes[p_idx].join() # Join processes

    C = matrix.reshape_2d(C[:], n_rows=N, n_cols=M)
    return C
