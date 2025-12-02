# fluid-gpu

## Problem definition

This code solves Burgers' partial differential equation on GPU using CUDA with a simple Python interface. Simple Burgers' equation can be written as:

$$
    \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial^2 x},
$$

where $u(x, t)$ is a function describing a field, and $\nu$ is a constant controlling diffusion. In multiple dimensions we get:

$$
    \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \nabla \mathbf{u} = \nu \nabla^2 \mathbf{u}.
$$

In this scenario $u$ can describe either scalar value at point $\mathbf{x}$ or be a multidimensional vector itself $\mathbf{u}$ defined at every point of higher dimensional domain. In this project we will solve the case of two dimensional burgers equation (where $\mathbf{u}$ is a vector field). We can represent $\mathbf{u}$ as two scalar componenst $u,v$ as:

$$
    \mathbf{u}(x,y,t) = \begin{bmatrix} u(x,y,t) \\\ v(x,y,t) \end{bmatrix}
$$

## Prerequisites

- Python 3.12
- NVIDIA GPU + CUDA toolkit (specifically the nvcc compiler)
- [uv](https://github.com/astral-sh/uv) package manager installed

## Setup

Clone the repository, sync using uv. 

```bash
git clone https://github.com/pawel002/fluid-gpu
cd fluid-gpu
uv sync
```

After activating the `.venv` you need to build and install the `fluig-gpu` package.

```bash
uv run python setup.py build_ext --inplace
uv pip install --link-mode=copy -e .
```

Verify installation by running simple test:

```bash
uv run python tests/test.py
```

### 2D solution details

The 2D Burgers' equation system in conservation form is:

$$
    \frac{\partial u}{\partial t} + \frac{\partial (u^2/2)}{\partial x} + \frac{\partial (uv)}{\partial y} = \nu \Big(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\Big)
$$

$$
    \frac{\partial v}{\partial t} + \frac{\partial (uv)}{\partial x} + \frac{\partial (v^2/2)}{\partial y} = \nu \Big(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\Big)
$$

We solve the update step using the Lax-Friedrichs (Rusanov) method. We describe the problem discretization with the following variables:

- $x_i = i \Delta x, i \in \{1, \cdots, N_x - 1\}$.
- $y_j = j \Delta x, j \in \{1, \cdots, N_y - 1\}$.
- $t^n = n \Delta t$.
- $u_{i,j}^n \approx u(x_i, y_j, t^n)$.
- $v_{i,j}^n \approx v(x_i, y_j, t^n)$.

#### 1. Flux Definitions

We define separate fluxes for the $u$ (horizontal) and $v$ (vertical) momentum equations to account for the conservative variable forms ($u^2/2$ vs $uv$).

**For the $u$-component equation:**

Flux along $x$-axis at $(i + \frac{1}{2}, j)$:
$$
    F_{i + 1 / 2,j}^{(u)} = \frac{1}{2}\left( \frac{(u_{i,j}^n)^2}{2} + \frac{(u_{i+1,j}^n)^2}{2} \right) - \frac{1}{2} \max(|u_{i,j}^n|, |u_{i+1,j}^n|)(u_{i+1,j}^n - u_{i,j}^n)
$$

Flux along $y$-axis at $(i, j + \frac{1}{2})$:
$$
    G_{i,j+1/2}^{(u)} = \frac{1}{2}(v_{i,j}^n u_{i,j}^n + v_{i,j+1}^n u_{i,j+1}^n) - \frac{1}{2} \max(|v_{i,j}^n|, |v_{i,j+1}^n|)(u_{i,j+1}^n - u_{i,j}^n)
$$

**For the $v$-component equation:**

Flux along $x$-axis at $(i + \frac{1}{2}, j)$:
$$
    F_{i + 1 / 2,j}^{(v)} = \frac{1}{2}(u_{i,j}^n v_{i,j}^n + u_{i+1,j}^n v_{i+1,j}^n) - \frac{1}{2} \max(|u_{i,j}^n|, |u_{i+1,j}^n|)(v_{i+1,j}^n - v_{i,j}^n)
$$

Flux along $y$-axis at $(i, j + \frac{1}{2})$:
$$
    G_{i,j+1/2}^{(v)} = \frac{1}{2}\left( \frac{(v_{i,j}^n)^2}{2} + \frac{(v_{i,j+1}^n)^2}{2} \right) - \frac{1}{2} \max(|v_{i,j}^n|, |v_{i,j+1}^n|)(v_{i,j+1}^n - v_{i,j}^n)
$$

#### 2. Diffusion and Update Step

The diffusion term is expressed using the standard 5-point Laplacian for $w \in \{u, v \}$:

$$
    (\nabla^2 w)_{i,j} \approx \Big(\frac{w_{i+1,j}^n - 2w_{i,j}^n + w_{i-1,j}^n}{\Delta x^2} + \frac{w_{i,j+1}^n - 2w_{i,j}^n + w_{i,j-1}^n}{\Delta y^2}\Big)
$$

Full update per cell $(i, j)$, for component $w$ (using the corresponding $F^{(w)}$ and $G^{(w)}$ defined above):

$$
\begin{split}
    w_{i,j}^{n+1} &= w_{i,j}^n \\
    &- \frac{\Delta t}{\Delta x} \Big(F^{(w)}_{i+1/2,j} - F^{(w)}_{i-1/2,j} \Big) \\
    &- \frac{\Delta t}{\Delta y} \Big(G^{(w)}_{i,j+1/2} - G^{(w)}_{i,j-1/2} \Big) \\
    &+ \nu \Delta t \Big[\frac{w_{i+1,j}^n - 2w_{i,j}^n + w_{i-1,j}^n}{\Delta x^2} + \frac{w_{i,j+1}^n - 2w_{i,j}^n + w_{i,j-1}^n}{\Delta y^2} \Big]
\end{split}
$$