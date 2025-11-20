import numpy as np
from fluidgpu import vector_add

A = np.array([1, 2, 3], dtype=np.float32)
B = np.array([10, 20, 30], dtype=np.float32)

C = vector_add(A, B)
print(C)