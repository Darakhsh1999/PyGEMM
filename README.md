# PyGEMM
PyGEMM - Python General Matrix Multiplication benchmarks

Benchmark of different matrix multiplication methods defined in <code>matmult.py</code>. The matrix multiplication is defined as $C = AB$ where $A,B$ are square matrices of size $(N,N)$.

![image1](matmul_bench.png)

Matrix multiplication using <code>threading</code> module. Computation time is independant of number of threads which is expected due to Python's global interpreter lock (GIL).

![image2](threading_bench.png)

Matrix multiplication using <code>multiprocessing</code> module. Computation time is increasing with increasing number of worker processes meaning overhead time outweighs the gains from parallelization.

![image3](multiprocess_bench.png)
