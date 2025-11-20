# fluid-gpu
This code solves Burger's partial differential equation on GPU using CUDA with simple python interface.

### Installation

To run virtual environment you need to have [UV](https://github.com/astral-sh/uv) package manager installed. Pull all neded libraries using:

```
uv sync
```

After activating the environment, you need to build the package on your architecture using:

```
uv run python setup.py build_ext --inplace
uv pip install -e .
```

Your virtual environment should now have the module `fluidgpu` loaded and can be used in other python files. Now you can try running `tests/test.py`.
