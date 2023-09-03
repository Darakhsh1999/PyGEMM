import random
import multiprocessing

def task(a,b,idx,zeros):
    zeros[idx] = a*b

if __name__ == "__main__":

    a,b = 1,5
    C = multiprocessing.Array("i", 5)
    n_processes = 4
    

    processes = []
    for p_idx in range(n_processes):
        a, b = random.randint(0,10), random.randint(0,10)
        print(a,b)
        p_i = multiprocessing.Process(target=task, args=(a, b, p_idx, C))
        p_i.start()
        processes.append(p_i)
    

    for p_idx in range(n_processes): processes[p_idx].join()
    print(C[:])
    
