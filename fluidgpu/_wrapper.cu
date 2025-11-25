#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include "vector_add.cuh"
#include "burgers_solver.cuh"

static PyObject* add_vectors(PyObject* self, PyObject* args)
{
    PyObject *A_obj = NULL;
    PyObject *B_obj = NULL;
    Py_ssize_t n_py = 0;

    // parse arguments
    if (!PyArg_ParseTuple(args, "OOn", &A_obj, &B_obj, &n_py)) {
        return NULL; 
    }

    npy_intp n = (npy_intp)n_py;

    // convert to right pointers
    PyArrayObject *A_arr = (PyArrayObject*) PyArray_FROM_OTF(A_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *B_arr = (PyArrayObject*) PyArray_FROM_OTF(B_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    // check for errors
    if (A_arr == NULL || B_arr == NULL) {
        Py_XDECREF(A_arr);
        Py_XDECREF(B_arr);
        return NULL;
    }

    // allocate output
    npy_intp dims[1] = { n };
    PyArrayObject *C_arr = (PyArrayObject*) PyArray_SimpleNew(1, dims, NPY_FLOAT);
    if (C_arr == NULL) {
        Py_DECREF(A_arr);
        Py_DECREF(B_arr);
        return NULL;
    }
    
    // raw pointers
    float *h_A = (float*) PyArray_DATA(A_arr);
    float *h_B = (float*) PyArray_DATA(B_arr);
    float *h_C = (float*) PyArray_DATA(C_arr);

    // addition on GPU
    int status = vector_add_cuda(h_A, h_B, h_C, n);

    // cleanup
    Py_DECREF(A_arr);
    Py_DECREF(B_arr);

    // check for error
    if (status != 0) {
        Py_DECREF(C_arr);
        PyErr_SetString(PyExc_RuntimeError, "CUDA vector_add failed");
        return NULL;
    }

    return (PyObject*) C_arr;
}

static PyObject* burgers_solver(PyObject* self, PyObject* args)
{
    PyObject *u_obj = NULL;
    PyObject *v_obj = NULL;
    
    int steps;
    float nu, dt, dx, dy;

    // parse objects
    if (!PyArg_ParseTuple(args, "OOiffff", 
        &u_obj, &v_obj, &steps, &nu, &dt, &dx, &dy)) {
        return NULL; 
    }

    // get data pointers
    PyArrayObject *u_arr = (PyArrayObject*)PyArray_FROM_OTF(u_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *v_arr = (PyArrayObject*)PyArray_FROM_OTF(v_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    if (u_arr == NULL || v_arr == NULL) {
        Py_XDECREF(u_arr);
        Py_XDECREF(v_arr);
        return NULL;
    }

    // get dimensions
    int nx = (int)PyArray_DIM(u_arr, 0);
    int ny = (int)PyArray_DIM(u_arr, 1);

    // get raw data pointers
    float *u = (float*)PyArray_DATA(u_arr);
    float *v = (float*)PyArray_DATA(v_arr);

    // call cuda wrapper
    int status = compute_burgers_steps(u, v, nx, ny, nu, dt, dx, dy, steps);

    // check status
    if (status != 0) {
        Py_DECREF(u_arr);
        Py_DECREF(v_arr);
        PyErr_SetString(PyExc_RuntimeError, "CUDA burgers solver failed");
        Py_RETURN_NONE;
    }

    // cleanup
    Py_DECREF(u_arr);
    Py_DECREF(v_arr);

    // return none
    Py_RETURN_NONE;
}

// method table
static PyMethodDef methods[] = {
    { "add_vectors",  (PyCFunction) add_vectors,    METH_VARARGS, "Elementwise addition of two 1D float32 arrays: C = A + B.\n" },
    { "solver_steps", (PyCFunction) burgers_solver, METH_VARARGS, "Solve 2D Burgers equation steps."},
    { NULL, NULL, 0, NULL }
};

// module definition
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_wrapper", "Fluid GPU solver core", -1, methods,
    NULL, NULL, NULL, NULL
};

// module initialization
PyMODINIT_FUNC 
PyInit__wrapper(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}