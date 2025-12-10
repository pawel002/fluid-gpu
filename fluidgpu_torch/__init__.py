import numpy as np
import torch

from .burgers_solver_torch import compute_burgers_steps


def solver_steps(
    u: np.ndarray,
    v: np.ndarray,
    nu: float,
    dt: float, 
    dx: float, 
    dy: float,
    steps: int = 1    
):
    """
    Advances the 2D Burgers' equation solver by a specified number of steps.
    
    Args:
        u (np.ndarray): 2D array of u-velocity components (float32).
        v (np.ndarray): 2D array of v-velocity components (float32).
        nu (float): Kinematic viscosity coefficient (must be positive).
        dt (float): Time step size.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        steps (int): Number of time steps to simulate.
    """

    assert isinstance(u, np.ndarray) and isinstance(v, np.ndarray), "Expected numpy arrays"
    assert u.shape == v.shape, "Expected arrays to have the same shape"
    assert (dim := len(u.shape)) == 2, f"Expected 2-D arrays, got {dim}-D arrays"

    assert isinstance(steps, int), "Steps needs to be int"
    assert steps >= 1, "Steps need to be >= 1"

    assert isinstance(nu, float), "nu needs to be a float"
    assert nu > 0 and nu < 1, "nu needs to be in the interval (0, 1)"

    params = {"dt": dt, "dx": dx, "dy": dy}
    for name, value in params.items():
        assert isinstance(value, float), f"{name} needs to be a float"
        assert value > 0, f"{name} needs to be > 0"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("using device:", device)

    u_torch = torch.from_numpy(np.ascontiguousarray(u, dtype=np.float32)).to(device)
    v_torch = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).to(device)

    compute_burgers_steps(
        u_torch, 
        v_torch, 
        float(nu),
        float(dt), 
        float(dx), 
        float(dy),
        steps
    )

    # Convert back to numpy and copy to original arrays
    u_result = u_torch.cpu().numpy()
    v_result = v_torch.cpu().numpy()
    
    np.copyto(u, u_result)
    np.copyto(v, v_result)