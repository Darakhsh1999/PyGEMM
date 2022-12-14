""" Matrix multiplication A*B, shape A [n,q], shape B [q,m] """
import numpy as np

def slice_2d(x, start_row, end_row, start_col, end_col): 
    return [row[start_col:end_col] for row in x[start_row:end_row]]

def assert_algorithm(C, C_truth):
    """ Compares own implementation to numpy (ground truth) """
    C_np = np.array(C)
    assert(C_np.shape == C_truth.shape)
    assert(np.less(np.abs(C_np-C_truth), 0.01).all())

def matmul_np(A, B):
    """ numpy internal matmul """
    return A @ B

def matmul_row_brute(A, B, C, n, q, m):
    """ Row major brute force """
    for row in range(n):
        for col in range(m):
            for k in range(q):
                C[row][col] += A[row][k] * B[k][col] 

def matmul_col_brute(A, B, C, n, q, m):
    """ Column major brute force """
    for col in range(m):
        for row in range(n):
            for k in range(q):
                C[row][col] += A[row][k] * B[k][col] 

def matmul_row_factored(A, B, C, n, m):
    """ Factored row vector"""
    for row in range(n):
        v = A[row]
        for col in range(m):
            for k, v_val in enumerate(v):
                C[row][col] += v_val * B[k][col] 

def matmul_block(A, B, C, n, p, m, k):
    """ Partition matrix into blocks """
    # A (n,p) : (q,s)
    # B (p,m) : (s,r)
    # q = n//k, s = p//k, r = m//k
    assert(n % k == 0)
    assert(m % k == 0)

    for q in range(0, n, k):
        for r in range(0, m, k):
            for s in range(0, p, k):
                block_A = slice_2d(A, q, q+k, s, s+k) # (k,k) 
                block_B = slice_2d(B, s, s+k, r, r+k) # (k,k) 

                for row in range(k):
                    for col in range(k):
                        for l in range(k):
                            C[q+row][r+col] += block_A[row][l] * block_B[l][col] 


def matmul_thread(A, B, n, q, m):
    """ Matrix mul using threads """
    pass