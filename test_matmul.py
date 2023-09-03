import matmul
import matrix
import numpy as np

if __name__ == "__main__":

    matrix_sizes = [
        (128,64,32),
        (64,64,64),
        (32,64,128),
    ]

    for (N,Q,M) in matrix_sizes:

        A = matrix.randint(N,Q)
        B = matrix.randint(Q,M)
        C_target = np.array(A) @ np.array(B) # ground truth

        A_lin = matrix.LinearMatrix((N,Q), "randint")
        B_lin = matrix.LinearMatrix((Q,M), "randint")
        C_target_lin = np.array(A_lin.to_matrix()) @ np.array(B_lin.to_matrix())

        # Row brute
        C = matmul.matmul_row_brute(A, B)
        matmul.assert_algorithm(C, C_target)

        # Col brute
        C = matmul.matmul_col_brute(A, B)
        matmul.assert_algorithm(C, C_target)

        # Row factored
        C = matmul.matmul_row_factored(A, B)
        matmul.assert_algorithm(C, C_target)

        # Col factored
        C = matmul.matmul_col_factored(A, B)
        matmul.assert_algorithm(C, C_target)

        # Factored
        C = matmul.matmul_factored(A, B)
        matmul.assert_algorithm(C, C_target)

        # Block 
        C = matmul.matmul_block(A, B, block_size=8)
        matmul.assert_algorithm(C, C_target)

        # Multithreading 
        C = matmul.matmul_thread(A, B, n_threads=4)
        matmul.assert_algorithm(C, C_target)

        # Multiprocessing 
        C = matmul.matmul_process(A, B, n_processes=4)
        matmul.assert_algorithm(C, C_target)

        # LinearMatrix row brute
        C_lin = matmul.matmul_linear_row_brute(A_lin, B_lin)
        matmul.assert_algorithm(C_lin.to_matrix(), C_target_lin)

        # LinearMatrix col brute
        C_lin = matmul.matmul_linear_col_brute(A_lin, B_lin)
        matmul.assert_algorithm(C_lin.to_matrix(), C_target_lin)

        # LinearMatrix row factored
        C_lin = matmul.matmul_linear_row_factored(A_lin, B_lin)
        matmul.assert_algorithm(C_lin.to_matrix(), C_target_lin)

        # LinearMatrix col factored
        C_lin = matmul.matmul_linear_col_factored(A_lin, B_lin)
        matmul.assert_algorithm(C_lin.to_matrix(), C_target_lin)

        # LinearMatrix factored
        C_lin = matmul.matmul_linear_factored(A_lin, B_lin)
        matmul.assert_algorithm(C_lin.to_matrix(), C_target_lin)

        # LinearMatrix block
        C_lin = matmul.matmul_linear_block(A_lin, B_lin, block_size=8)
        matmul.assert_algorithm(C_lin.to_matrix(), C_target_lin)

        # LinearMatrix multiprocessing
        C_lin = matmul.matmul_linear_process(A_lin, B_lin, n_processes=4)
        matmul.assert_algorithm(C_lin, C_target_lin)

        print(f"Test passed for (N,Q,M) = {(N,Q,M)}")
