#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

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
    float *A_data = (float*) PyArray_DATA(A_arr);
    float *B_data = (float*) PyArray_DATA(B_arr);
    float *C_data = (float*) PyArray_DATA(C_arr);

    // addition
    for (npy_intp i = 0; i < n; ++i) {
        C_data[i] = A_data[i] + B_data[i];
    }

    // cleanup
    Py_DECREF(A_arr);
    Py_DECREF(B_arr);
    return (PyObject*) C_arr;
}

// method table
static PyMethodDef methods[] = {
    { "add_vectors", (PyCFunction) add_vectors, METH_VARARGS, "Elementwise addition of two 1D float32 arrays: C = A + B.\n" },
    { NULL, NULL, 0, NULL }
};

// module definition
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_solver", "Fluid GPU solver core", -1, methods,
    NULL, NULL, NULL, NULL
};

// module initialization
PyMODINIT_FUNC 
PyInit__solver(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}