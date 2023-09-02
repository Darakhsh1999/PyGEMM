import matmul
import matrix
import numpy as np

N = 128
Q = 64
M = 64

A = matrix.randint(N,Q)
B = matrix.randint(Q,M)
A_lin = matrix.LinearMatrix((N,Q), "randint")
B_lin = matrix.LinearMatrix((Q,M), "randint")

C_target = np.array(A) @ np.array(B) # ground truth
C_target_lin = np.array(A_lin.to_matrix()) @ np.array(B_lin.to_matrix())


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

# LinearMatrix brute
C_lin = matrix.LinearMatrix((N,M), "zeros")
matmul.matmul_linear_brute(A_lin, B_lin, C_lin)
matmul.assert_algorithm(C_lin.to_matrix(), C_target_lin)

print("Passed all tests!")
