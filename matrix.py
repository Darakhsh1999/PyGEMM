import random

def zeros(n_rows: int, n_cols: int):
    """ Zero matrix list comprehension (n_row, n_cols) """
    return [[0 for _ in range(n_cols)] for _ in range(n_rows)]

def zeros_brute(n_rows: int, n_cols: int):
    """ Zero matrix brute force (n_row, n_cols) """
    matrix = []
    for _ in range(n_rows):
        for _ in range(n_cols):
            matrix.append(0)
    return matrix

def randint(n_rows: int, n_cols: int, low: int = 0, high: int = 1024):
    """ Random integer matrix using python lists (n_row, n_cols) """
    return [[random.randint(low, high) for _ in range(n_cols)] for _ in range(n_rows)]

def rand(n_rows: int, n_cols: int):
    """ Random float matrix using python lists (n_row, n_cols) """
    return [[random.random() for _ in range(n_cols)] for _ in range(n_rows)]
