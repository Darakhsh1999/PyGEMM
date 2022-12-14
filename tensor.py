""" Tensor class with functionaly somewhere between numpy and python lists. """

# slicing 
# https://stackoverflow.com/questions/3680262/how-to-slice-a-2d-python-array-fails-with-typeerror-list-indices-must-be-int

class tensor():

    def __init__(self, x: list):
        
        self.x = x # tensor
        if x:
            self.shape = self.get_shape(x)
            self.dimension = len(self.shape)
            self.n_elements = self.get_numel()
        else:
            self.shape = []
            self.dimension = 0
            self.n_elements = 0

   
    def get_shape(self, reduced_list: list):
        """ Returns a tuple of tensor shape """

        if not isinstance(reduced_list, list):
            return []
        return [len(reduced_list)] + self.get_shape(reduced_list[0])

    def get_numel(self):
        """ Returns number of elements in tensor """

        n_elements = 1
        if self.shape is not None:
            for factor in self.shape:
                n_elements *= factor 
            return  n_elements
        else:
            return 0


    def slice_2d(self, start_row, end_row, start_col, end_col): # move to another module
        return [row[start_col:end_col] for row in self.x[start_row:end_row]]

    def __str__(self):
        return str(self.x)

    def __getitem__(self, input):
        """ Slice syntax implementation"""
        
        print("Recieved input type", type(input))
        print(input)

        if not self.x:
            return []

        if isinstance(input, int):
            if input < 0:
                input += self.shape[0]
            return self.x[input] 
        elif isinstance(input, list):
            return [self.x[inner_dim_idx] for inner_dim_idx in input]
        elif isinstance(input, slice):
            start = input.start            
            stop = input.stop
            step = input.step
            if start is None:
                start = 0
            elif start < 0:
                start += self.shape[0]
            if stop is None:
                stop = self.shape[0]
            elif stop < 0:
                stop += self.shape[0]
            if step is None:
                step = 1
            return [self.x[inner_dim_idx] for inner_dim_idx in range(start, stop, step)] # add index bound check
        elif isinstance(input, tuple):
            
            if len(input) > self.dimension: # too many indices
                raise ValueError("Index to high dimension")

            dim = 0 
            # first dimension
            if isinstance(input[dim], int):
                if input[dim] < 0:
                    input[dim] += self.shape[dim]
                q = self.x[input[dim]]
            elif isinstance(input[dim], list):
                q = [self.x[inner_dim_idx] for inner_dim_idx in input[dim]]
            elif isinstance(input[dim], slice): # slice
                start = input[dim].start            
                stop = input[dim].stop
                step = input[dim].step
                if start is None:
                    start = 0
                elif start < 0:
                    start += self.shape[dim]
                if stop is None:
                    stop = self.shape[dim]
                elif stop < 0:
                    stop += self.shape[dim]
                if step is None:
                    step = 1
                return [self.x[inner_dim_idx] for inner_dim_idx in range(start, stop, step)] # add index bound check
            else:
                raise TypeError("Received unexpected indexing type.")


            # rest of dimensions
            for dim in range(1, len(input)): # rest of dimensions

                if isinstance(input[dim], int):
                    if input[dim] < 0:
                        input[dim] += self.shape[dim]
                    q = q[input[dim]]
                elif isinstance(input[dim], list):
                    q = [q[inner_dim_idx] for inner_dim_idx in input[dim]]
                elif isinstance(input[dim], slice): # slice
                    start = input[dim].start            
                    stop = input[dim].stop
                    step = input[dim].step
                    if start is None:
                        start = 0
                    elif start < 0:
                        start += self.shape[dim]
                    if stop is None:
                        stop = self.shape[dim]
                    elif stop < 0:
                        stop += self.shape[dim]
                    if step is None:
                        step = 1
                    return [q[inner_dim_idx] for inner_dim_idx in range(start, stop, step)] # add index bound check
                else:
                    raise TypeError("Received unexpected indexing type.")
            return q
        else:
            raise RuntimeError("Unexpected indexing")


if __name__ == '__main__':


    vector = [1,2,3,4,5]
    matrix = [ # (4,3)
        [1,2,3],
        [4,5,6],
        [7,8,9],
        [10,11,12]
        ]
    tensor3D = [ # (3,2,2)
        [[1,2],[3,4]],
        [[5,6],[7,8]],
        [[9,10],[11,12]]
        ]
    import numpy as np
    tensor_object = tensor(tensor3D)
    tensor_np = np.array(tensor3D)
    print(tensor_np[0,:,1])