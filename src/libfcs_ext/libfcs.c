#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <fcs.h>


static PyObject* loadFCS(PyObject *self, PyObject *args) {
    PyObject* filename_bytes;
    char* filename;
    Py_ssize_t filename_len;
    if (!PyArg_ParseTuple(args, "O&", PyUnicode_FSConverter, &filename_bytes))
    {
        return NULL;
    }

    PyBytes_AsStringAndSize(filename_bytes, &filename, &filename_len);
    FCSFile* fcs = load_FCS(filename);

    printf("Loaded FCS file with %zd parameters and %zd events\n", fcs->compensated.n_parameters, fcs->compensated.n_events);
    if (fcs->metadata.comment.present) {
        printf("\t%.*s\n",
                fcs->metadata.comment.string.length,
                fcs->metadata.comment.string.buffer
        );
    }
    printf("\tmode=%d\n\tdatatype=%d\n", fcs->metadata.mode, fcs->metadata.datatype);
    // Print the parameters
    printf("\tParameters:\n");
    for (int i = 0; i < fcs->metadata.n_parameters; ++i) {
        printf("\t\t- %.*s",
                fcs->metadata.parameters[i].short_name.length,
                fcs->metadata.parameters[i].short_name.buffer
        );
        if (fcs->metadata.parameters[i].name.present) {
            printf(" (%.*s)", 
                    fcs->metadata.parameters[i].name.string.length,
                    fcs->metadata.parameters[i].name.string.buffer
            );
        }
        printf("\n");
    }
    // Print the first five events
    printf("\tEvents:\n");
    for (int i = 0; i < 5; ++i) {
        printf("\t");
        for (int j = 0; j < fcs->compensated.n_parameters; ++j) {
            printf("\t%.2e", fcs->compensated.data[(i * fcs->compensated.n_parameters) + j]);
        }
        printf("\n");
    }

    Py_DECREF(filename_bytes);
    Py_RETURN_NONE;
}

static PyMethodDef FCSMethods[] = {
    {"loadFCS", loadFCS, METH_VARARGS, "Loads an FCS file"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef libfcsmodule = {
    PyModuleDef_HEAD_INIT,
    "_libfcs_ext",
    NULL,
    -1,
    FCSMethods
};

PyMODINIT_FUNC
PyInit__libfcs_ext(void)
{
    if (libfcs_init())
        return PyModule_Create(&libfcsmodule);
    return NULL;
}