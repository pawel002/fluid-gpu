#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include "burgers_solver.cuh"

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
    { "solver_steps", (PyCFunction) burgers_solver, METH_VARARGS, "Solve 2D Burgers equation steps"},
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