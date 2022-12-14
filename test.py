import matrix
import matmul
import numpy as np

def slice_2d(x, start_row, end_row, start_col, end_col): # move to another module
    return [row[start_col:end_col] for row in x[start_row:end_row]]

N = 4 
k = 2
A = matrix.randint(N,N)
B = matrix.randint(N,N)
C = matrix.zeros(N,N)
C_real = np.array(A) @ np.array(B)
matmul.matmul_block(A, B, C, N, N, N, k)
matmul.assert_algorithm(C, C_real)


# q = n//k, s = p//k, r = m//k
# A (n,p) : (q,s)
# B (p,m) : (s,r)
