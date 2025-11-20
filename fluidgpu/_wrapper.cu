#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include "vector_add.cuh"

static PyObject* add_vectors(PyObject* self, PyObject* args)
{
    PyObject *A_obj = NULL;
    PyObject *B_obj = NULL;
    Py_ssize_t n_py = 0;

    // parse arguments: two Python objects
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

// method table
static PyMethodDef methods[] = {
    { "add_vectors", (PyCFunction) add_vectors, METH_VARARGS, "Elementwise addition of two 1D float32 arrays: C = A + B.\n" },
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