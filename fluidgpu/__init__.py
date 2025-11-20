import numpy as np
from fluidgpu import _wrapper as _cwrapper

def vector_add(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray), "Expected numpy arrays"
    assert A.shape == B.shape, "Expected arrays to have the same shape"
    assert (dim := len(A.shape)) == 1, f"Expected 1-D arrays, got {dim}-D arrays"

    n = A.shape[0]
    return _cwrapper.add_vectors(
        A.astype(np.float32),
        B.astype(np.float32),
        n,
    )
