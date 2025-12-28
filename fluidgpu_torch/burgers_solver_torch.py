import torch


@torch.jit.script
def burgers_step(u: torch.Tensor, v: torch.Tensor, u_new: torch.Tensor, v_new: torch.Tensor, 
                 cx: float, cy: float, diff_coef: float, inv_dx2: float, inv_dy2: float):
    u_C = u[1:-1, 1:-1]  
    u_L = u[1:-1, :-2]   
    u_R = u[1:-1, 2:]    
    u_T = u[:-2, 1:-1]   
    u_B = u[2:, 1:-1]    
    
    v_C = v[1:-1, 1:-1]  
    v_L = v[1:-1, :-2]   
    v_R = v[1:-1, 2:]    
    v_T = v[:-2, 1:-1]   
    v_B = v[2:, 1:-1]    
    
    # 1. horizontal Fluxes (F) for U-equation
    max_u_R = torch.maximum(torch.abs(u_C), torch.abs(u_R))
    flux_F_R_u = 0.5 * (0.5 * (u_C * u_C + u_R * u_R)) - 0.5 * max_u_R * (u_R - u_C)
    
    max_u_L = torch.maximum(torch.abs(u_L), torch.abs(u_C))
    flux_F_L_u = 0.5 * (0.5 * (u_L * u_L + u_C * u_C)) - 0.5 * max_u_L * (u_C - u_L)
    
    # 2. vertical Fluxes (G) for U-equation
    max_v_B = torch.maximum(torch.abs(v_C), torch.abs(v_B))
    flux_G_B_u = 0.5 * (v_C * u_C + v_B * u_B) - 0.5 * max_v_B * (u_B - u_C)
    
    max_v_T = torch.maximum(torch.abs(v_T), torch.abs(v_C))
    flux_G_T_u = 0.5 * (v_T * u_T + v_C * u_C) - 0.5 * max_v_T * (u_C - u_T)
    
    # 3. horizontal Fluxes (F) for V-equation
    flux_F_R_v = 0.5 * (u_C * v_C + u_R * v_R) - 0.5 * max_u_R * (v_R - v_C)
    flux_F_L_v = 0.5 * (u_L * v_L + u_C * v_C) - 0.5 * max_u_L * (v_C - v_L)
    
    # 4. vertical Fluxes (G) for V-equation
    flux_G_B_v = 0.5 * (0.5 * (v_C * v_C + v_B * v_B)) - 0.5 * max_v_B * (v_B - v_C)
    flux_G_T_v = 0.5 * (0.5 * (v_T * v_T + v_C * v_C)) - 0.5 * max_v_T * (v_C - v_T)
    
    # 5. diffusion (5 point laplace)
    lap_u = (u_R - 2.0 * u_C + u_L) * inv_dx2 + (u_B - 2.0 * u_C + u_T) * inv_dy2
    lap_v = (v_R - 2.0 * v_C + v_L) * inv_dx2 + (v_B - 2.0 * v_C + v_T) * inv_dy2
    
    # 6. update
    u_new[1:-1, 1:-1] = u_C - cx * (flux_F_R_u - flux_F_L_u) \
                                  - cy * (flux_G_B_u - flux_G_T_u) \
                                  + diff_coef * lap_u
    
    v_new[1:-1, 1:-1] = v_C - cx * (flux_F_R_v - flux_F_L_v) \
                                  - cy * (flux_G_B_v - flux_G_T_v) \
                                  + diff_coef * lap_v


def compute_burgers_steps(u, v, nu, dt, dx, dy, steps):
    device = u.device
    dtype = u.dtype
    
    dx2 = dx * dx
    dy2 = dy * dy
    inv_dx2 = 1.0 / dx2
    inv_dy2 = 1.0 / dy2
    cx = dt / dx
    cy = dt / dy
    diff_coef = nu * dt
    
    # Pre-allocate buffers for ping-pong (like CUDA version)
    u_curr = u
    v_curr = v
    u_next = torch.zeros_like(u)
    v_next = torch.zeros_like(v)
    
    # Main loop with buffer swapping
    for k in range(steps):
        burgers_step(u_curr, v_curr, u_next, v_next, cx, cy, diff_coef, inv_dx2, inv_dy2)
        
        # Swap buffers
        u_curr, u_next = u_next, u_curr
        v_curr, v_next = v_next, v_curr
    
    # Copy final result back to input tensors if needed
    if u_curr is not u:
        u.copy_(u_curr)
    if v_curr is not v:
        v.copy_(v_curr)
