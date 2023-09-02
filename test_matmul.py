import matmul
import matrix
import numpy as np

N = 128
Q = 64
M = 64

A = matrix.randint(N,Q)
B = matrix.randint(Q,M)

C_target = np.array(A) @ np.array(B) # ground truth


# Row brute
C = matrix.zeros(N,M)
matmul.matmul_row_brute(A, B, C)
matmul.assert_algorithm(C, C_target)

# Col brute
C = matrix.zeros(N,M)
matmul.matmul_col_brute(A, B, C)
matmul.assert_algorithm(C, C_target)

# Row factored
C = matrix.zeros(N,M)
matmul.matmul_row_factored(A, B, C)
matmul.assert_algorithm(C, C_target)

# Col factored
C = matrix.zeros(N,M)
matmul.matmul_col_factored(A, B, C)
matmul.assert_algorithm(C, C_target)

# Factored
C = matrix.zeros(N,M)
matmul.matmul_factored(A, B, C)
matmul.assert_algorithm(C, C_target)

# Block 
C = matrix.zeros(N,M)
matmul.matmul_block(A, B, C, block_size=8)
matmul.assert_algorithm(C, C_target)

# Threading 
C = matrix.zeros(N,M)
matmul.matmul_thread(A, B, C, n_threads=4)
matmul.assert_algorithm(C, C_target)

print("Passed all tests!")
