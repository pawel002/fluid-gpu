# fluid-gpu
This code solves Burger's partial differential equation on GPU using CUDA with a simple Python interface.

## Prerequisites

- Python 3.12
- C compiler (e.g. gcc / clang)
- [uv](https://github.com/astral-sh/uv) package manager installed
- (TODO) NVIDIA GPU + CUDA toolkit (see CUDA section)

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
uv pip install -e .
```

Verify installation by running simple test:

```bash
uv run python tests/test.py
```
