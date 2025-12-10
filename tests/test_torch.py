import numpy as np
from fluidgpu_torch import solver_steps


nx, ny = 128, 128

nu = 0.01
dx = 1.0 / (nx - 1)
dy = 1.0 / (ny - 1)
dt = 0.001
steps = 100

u = np.zeros((ny, nx), dtype=np.float32)
v = np.zeros((ny, nx), dtype=np.float32)

x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
X, Y = np.meshgrid(x, y, indexing="xy")

u += np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01).astype(np.float32)
v += 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02).astype(np.float32)

print("Before:")
print("u min/max:", u.min(), u.max())
print("v min/max:", v.min(), v.max())

solver_steps(
    u=u,
    v=v,
    nu=float(nu),
    dt=float(dt),
    dx=float(dx),
    dy=float(dy),
    steps=steps,
)

print("\nAfter:")
print("u min/max:", u.min(), u.max())
print("v min/max:", v.min(), v.max())


