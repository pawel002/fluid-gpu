# fluid-gpu
This code solves Burgers' partial differential equation on GPU using CUDA with a simple Python interface. Simple Burgers' equation can be written as:

$$
    \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial^2 x},
$$

where $u(x, t)$ is a function describing a field. In multiple dimensions we get:

$$
    \frac{\partial u}{\partial t} + u \nabla u = \nu \nabla^2 u.
$$

In this scenario $u$ can describe either scalar value at point $\mathbf{x}$ or be a multidimensional vector itself $\mathbf{u}$ defined at every point of higher dimensional domain.


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
