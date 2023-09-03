import math
import random

### Static matrices
def zeros(n_rows: int, n_cols: int):
    """ Zero matrix (n_row, n_cols) """
    return [[0 for _ in range(n_cols)] for _ in range(n_rows)]

def ones(n_rows: int, n_cols: int):
    """ One matrix (n_row, n_cols) """
    return [[1 for _ in range(n_cols)] for _ in range(n_rows)]

def fill(n_rows: int, n_cols: int, q):
    """ Fill matrix with value q (n_row, n_cols) """
    return [[q for _ in range(n_cols)] for _ in range(n_rows)]

### Random matrices
def randint(n_rows: int, n_cols: int, low: int = 0, high: int = 1024):
    """ Random integer matrix using python lists (n_row, n_cols) """
    return [[random.randint(low, high) for _ in range(n_cols)] for _ in range(n_rows)]

def rand(n_rows: int, n_cols: int):
    """ Random float matrix using python lists (n_row, n_cols) """
    return [[random.random() for _ in range(n_cols)] for _ in range(n_rows)]

### Helper functions

def slice_2d(x, start_row, end_row, start_col, end_col): 
    return [row[start_col:end_col] for row in x[start_row:end_row]]

def get_col(x, col_idx):
    return [row[col_idx] for row in x]

### Linear matrix
class LinearMatrix():
    """ Matrix stored in linear memory """

    def __init__(self, shape, type="zero"):
        assert len(shape) == 2
        self.n_rows, self.n_cols = shape
        self.N = self.n_rows * self.n_cols
        self.min, self.max = None, None
        self.initialize(type)
    
    def initialize(self, type):

        if type == "zeros":
            self.memory = self.N * [0]
        elif type == "ones":
            self.memory = self.N * [1]
        elif type == "randint":
            self.memory = [random.randint(0, 1024) for _ in range(self.N)]
        elif type == "rand":
            self.memory = [random.random() for _ in range(self.N)]
        elif isinstance(type, (int,float)):
            self.memory = self.N * [type]
        else:
            raise ValueError(f"Unexpected type {type}")
    
    def max_min(self):
        """ Find max and min value in matrix """
        min, max = math.inf, -math.inf
        for x in self.memory:
            if x < min: min = x
            if x > max: max = x
        
        self.min, self.max = min, max
    
    def to_matrix(self):
        """ Reshapes linear matrix to standard 2D list nested matrix"""
        return [[self.memory[row*self.n_cols+col] for col in range(self.n_cols)] for row in range(self.n_rows)]
    
    def reshape(self, shape):
        """ General version of to_matrix() """
        pass
    
    def __getitem__(self, idx):
        return self.memory[idx]
    
    def __setitem__(self, idx, val):
        self.memory[idx] = val
    
    def __str__(self) -> str:
        if (self.min is None) or (self.max is None): self.max_min()
        w = max(len(str(self.min)), len(str(self.max))) # width
        return "[" + "\n ".join( ["["+" ".join( [(f"{x:.4f}" if isinstance(x, float) else f"{x:{w}}") for x in self.memory[i*self.n_cols:(1+i)*self.n_cols]])+"]" for i in range(self.n_rows)] ) + "]"