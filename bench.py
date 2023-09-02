import time
from tqdm import tqdm

def bench(fun, n_iterations, *args):
    """ Benchmark function, returns timings: (t_tot,t_iter) """

    t0 = time.time()
    for _ in range(n_iterations):
        fun(*args)
    t_tot = time.time() - t0
    return t_tot, t_tot/n_iterations
