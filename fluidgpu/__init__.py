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

def solver_steps(
    u: np.ndarray,
    v: np.ndarray,
    nu: float,
    dt: float, 
    dx: float, 
    dy: float,
    steps: int = 1    
):
    assert isinstance(u, np.ndarray) and isinstance(v, np.ndarray), "Expected numpy arrays"
    assert u.shape == v.shape, "Expected arrays to have the same shape"
    assert (dim := len(u.shape)) == 2, f"Expected 2-D arrays, got {dim}-D arrays"

    assert isinstance(steps, int), "Number of steps needs to be int"
    assert steps >= 1, "Number of steps need to be >= 1"

    assert isinstance(nu, float), "nu needs to be a float"
    assert nu > 0 and nu < 1, "nu needs to be in the interval (0, 1)"

    for variable in [dt, dx, dy]:
        assert isinstance(variable, float), f"{variable.__name__} needs to be a float"
        assert variable > 0, f"{variable.__name__} needs to be > 0"

    _cwrapper.solver_steps(u, v, steps, nu, dt, dx, dy)