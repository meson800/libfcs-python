#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <math.h>

#include <Python.h>
#include "numpy/npy_common.h"
#include "numpy/ndarrayobject.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#include <fcs.h>
#include "logicle.h"
#include "hyperlog.h"

typedef struct {
    PyObject_HEAD
    /* Type-specific fields */
    FCSFile* file;
} FCSObject;

static PyObject * FCSObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"filename", NULL};
    FCSObject *self;
    PyObject *filename_bytes;

    self = (FCSObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        // Convert passed filename to string, in case it is a Path.
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, PyUnicode_FSConverter, &filename_bytes)) {
            return NULL;
        }
        char* filename;
        Py_ssize_t filename_len;
        if (PyBytes_AsStringAndSize(filename_bytes, &filename, &filename_len) == -1) {
            Py_DECREF(filename_bytes);
            return NULL;
        }
        self->file = load_FCS(filename);

        Py_DECREF(filename_bytes);
    }
    return (PyObject *) self;
}

static void FCSObject_dealloc(FCSObject *self) {
    free_FCS(self->file);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject* FCSObject_get_n_events_aborted(FCSObject *self, void *_closure) {
    if (self->file->metadata.n_events_aborted.present) {
        return Py_BuildValue("L", self->file->metadata.n_events_aborted.value);
    }
    Py_RETURN_NONE;
}

static PyObject* get_optional_string(OptionalString* ostr, const char* error_msg) {
    if (ostr->present) {
        PyObject* str = PyUnicode_DecodeUTF8(
            ostr->string.buffer, ostr->string.length, "strict"
        );
        if (str == NULL) {
            PyErr_SetString(PyExc_ValueError, error_msg);
        }
        return str;
    }
    Py_RETURN_NONE;
}

static PyObject* FCSObject_get_acquire_time(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.acquire_time, "Unable to decode acquisition time");
}

static PyObject* FCSObject_get_acquire_end_time(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.acquire_end_time, "Unable to decode acquisition end time");
}

static PyObject* FCSObject_get_acquire_date(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.acquire_date, "Unable to decode acquisition date");
}

static PyObject* FCSObject_get_cells(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.cells, "Unable to decode cell description");
}

static PyObject* FCSObject_get_comment(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.comment, "Unable to decode comment");
}

static PyObject* FCSObject_get_cytometer_type(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.cytometer_serial_number, "Unable to decode cytometer type");
}

static PyObject* FCSObject_get_cytometer_sn(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.cytometer_serial_number, "Unable to decode cytometer serial number");
}

static PyObject* FCSObject_get_institution(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.institution, "Unable to decode institution");
}

static PyObject* FCSObject_get_experimenter(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.experimenter, "Unable to decode experimenter name");
}

static PyObject* FCSObject_get_operator(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.operator, "Unable to decode operator name");
}

static PyObject* FCSObject_get_original_filename(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.filename, "Unable to decode original filename");
}

static PyObject* FCSObject_get_last_modified(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.last_modified, "Unable to decode last modified time");
}

static PyObject* FCSObject_get_last_modifier(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.last_modifier, "Unable to decode last modifier name");
}

static PyObject* FCSObject_get_plate_id(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.plate_id, "Unable to decode plate ID");
}

static PyObject* FCSObject_get_plate_name(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.plate_name, "Unable to decode plate name");
}

static PyObject* FCSObject_get_project(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.project, "Unable to decode project name");
}

static PyObject* FCSObject_get_specimen(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.specimen, "Unable to decode specimen");
}

static PyObject* FCSObject_get_specimen_source(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.specimen_source, "Unable to decode specimen source");
}

static PyObject* FCSObject_get_computer(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.computer, "Unable to decode acquisition computer name");
}

static PyObject* FCSObject_get_well_id(FCSObject *self, void *_closure) {
    return get_optional_string(&self->file->metadata.well_id, "Unable to decode well id");
}

static PyObject* FCSObject_get_uncompensated(FCSObject *self, void *_closure) {
    npy_intp dimensions[2] = {self->file->uncompensated.n_rows, self->file->uncompensated.n_cols};
    PyObject* np_array = PyArray_SimpleNewFromData(2, dimensions, NPY_DOUBLE, self->file->uncompensated.data);
    if (np_array == NULL) {
        return NULL;
    }
    Py_INCREF(self);
    if (PyArray_SetBaseObject(np_array, self) != 0) {
        Py_DECREF(self);
        Py_DECREF(np_array);
        return NULL;
    }
    return np_array;
}

static PyObject* FCSObject_get_compensated(FCSObject *self, void *_closure) {
    npy_intp dimensions[2] = {self->file->compensated.n_rows, self->file->compensated.n_cols};
    PyObject* np_array = PyArray_SimpleNewFromData(2, dimensions, NPY_DOUBLE, self->file->compensated.data);
    if (np_array == NULL) {
        return NULL;
    }
    Py_INCREF(self);
    if (PyArray_SetBaseObject(np_array, self) != 0) {
        Py_DECREF(self);
        Py_DECREF(np_array);
        return NULL;
    }
    return np_array;
}

static PyGetSetDef FCSObject_getsetters[] = {
        {"uncompensated", (getter) FCSObject_get_uncompensated, NULL,
            "Uncompensated events", NULL},
        {"compensated", (getter) FCSObject_get_compensated, NULL,
            "Compensated events", NULL},
        {"n_events_aborted", (getter) FCSObject_get_n_events_aborted, NULL,
            "Number of aborted events", NULL},
        {"acquire_time_str", (getter) FCSObject_get_acquire_time, NULL,
            "Acquisition start time as a string", NULL},
        {"acquire_end_time_str", (getter) FCSObject_get_acquire_end_time, NULL,
            "Acquisition end time as a string", NULL},
        {"acquire_date", (getter) FCSObject_get_acquire_date, NULL,
            "Acquisition date as a string", NULL},
        {"cells", (getter) FCSObject_get_cells, NULL,
            "Cell description", NULL},
        {"comment", (getter) FCSObject_get_comment, NULL,
            "User-added comment", NULL},
        {"cytometer_type", (getter) FCSObject_get_cytometer_type, NULL,
            "Type of flow cytometer", NULL},
        {"cytometer_serial_number", (getter) FCSObject_get_cytometer_sn, NULL,
            "Serial number of the acquisition flow cytometer", NULL},
        {"institution", (getter) FCSObject_get_institution, NULL,
            "Institution", NULL},
        {"experimenter", (getter) FCSObject_get_experimenter, NULL,
            "Experimenter name", NULL},
        {"operator", (getter) FCSObject_get_operator, NULL,
            "Instrument operator name", NULL},
        {"original_filename", (getter) FCSObject_get_original_filename, NULL,
            "Original FCS filename", NULL},
        {"last_modified_str", (getter) FCSObject_get_last_modified, NULL,
            "Datetime of last modification as a string", NULL},
        {"last_modifier", (getter) FCSObject_get_last_modifier, NULL,
            "Name of the last modifier", NULL},
        {"plate_id", (getter) FCSObject_get_plate_id, NULL,
            "Plate identifier", NULL},
        {"plate_name", (getter) FCSObject_get_plate_name, NULL,
            "Name of the plate", NULL},
        {"project", (getter) FCSObject_get_project, NULL,
            "Project name", NULL},
        {"specimen", (getter) FCSObject_get_specimen, NULL,
            "Specimen name", NULL},
        {"specimen_source", (getter) FCSObject_get_specimen_source, NULL,
            "Specimen source", NULL},
        {"computer", (getter) FCSObject_get_computer, NULL,
            "Name of the acquisition computer", NULL},
        {"well_id", (getter) FCSObject_get_well_id, NULL,
            "Well identifier", NULL},
    {NULL} /* Sentinel */
};

static PyTypeObject FCSType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_libfcs_ext.FCS",
    .tp_doc = PyDoc_STR("FCS object"),
    .tp_basicsize = sizeof(FCSObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_FINALIZE,
    .tp_new = FCSObject_new,
    .tp_dealloc = FCSObject_dealloc,
    .tp_getset = FCSObject_getsetters,
};

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

    printf("Loaded FCS file with %zd parameters and %zd events\n", fcs->compensated.n_cols, fcs->compensated.n_rows);
    if (fcs->metadata.comment.present) {
        printf("\t%.*s\n",
                fcs->metadata.comment.string.length,
                fcs->metadata.comment.string.buffer
        );
    }
    printf("\tmode=%d\n\tdatatype=%d\n", fcs->metadata.mode, fcs->metadata.datatype);
    // Print the parameters
    puts("\tParameters:");
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
        puts("");
    }
    // Print the first five events
    puts("\tEvents:\n");
    for (int i = 0; i < 5; ++i) {
        printf("%s", "\t");
        for (int j = 0; j < fcs->compensated.n_cols; ++j) {
            printf("\t%.2e", fcs->compensated.data[(i * fcs->compensated.n_cols) + j]);
        }
        printf("%s", "\n");
    }

    // Print extra keyvals

    printf("\n\t%d extra keyvals:\n", fcs->metadata.extra_keyvals.n_vals);
    for (int i = 0; i < fcs->metadata.extra_keyvals.n_vals; ++i) {
        printf("\t\t%.*s: %.*s\n",
            fcs->metadata.extra_keyvals.items[i].key.length,
            fcs->metadata.extra_keyvals.items[i].key.buffer,
            fcs->metadata.extra_keyvals.items[i].value.length,
            fcs->metadata.extra_keyvals.items[i].value.buffer
        );
    }

    Py_DECREF(filename_bytes);


    Py_RETURN_NONE;
}

/************** ufuncs for fast Numpy processing *******************/
#define TO_D(x) (*((double*)x))
static void double_flin(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    npy_intp n = dimensions[0];
    char *in = args[0], *in_T = args[1], *in_A = args[2];
    char *out = args[3];
    npy_intp in_step = steps[0], in_T_step = steps[1], in_A_step = steps[2];
    npy_intp out_step = steps[3];

    for (npy_intp i = 0; i < n; ++i) {
        TO_D(out) = (TO_D(in) + TO_D(in_A)) / (TO_D(in_T) + TO_D(in_A));

        in += in_step;
        in_T += in_T_step;
        in_A += in_A_step;
        out += out_step;
    }
}
PyUFuncGenericFunction flin_func[1] = {&double_flin};
static char flin_types[4] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static void double_flog(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    npy_intp n = dimensions[0];
    char *in = args[0], *in_T = args[1], *in_M = args[2];
    char *out = args[3];
    npy_intp in_step = steps[0], in_T_step = steps[1], in_M_step = steps[2];
    npy_intp out_step = steps[3];

    for (npy_intp i = 0; i < n; ++i) {
        // Check for out of bounds (x <= 0) and return NaN
        if (TO_D(in) <= 0) {
            TO_D(out) = NPY_NAN;
        } else {
            TO_D(out) = 1 + log10(TO_D(in) / TO_D(in_T)) / TO_D(in_M);
        }

        in += in_step;
        in_T += in_T_step;
        in_M += in_M_step;
        out += out_step;
    }
}
PyUFuncGenericFunction flog_func[1] = {&double_flog};
static char flog_types[4] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static void double_fasinh(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    npy_intp n = dimensions[0];
    char *in = args[0], *in_T = args[1], *in_M = args[2], *in_A = args[3];
    char *out = args[4];
    npy_intp in_step = steps[0], in_T_step = steps[1], in_M_step = steps[2], in_A_step = steps[3];
    npy_intp out_step = steps[4];

    double ln_10 = log(10.0);

    for (npy_intp i = 0; i < n; ++i) {
        TO_D(out) = (
            asinh(TO_D(in) * sinh(TO_D(in_M) * ln_10) / TO_D(in_T))
            + TO_D(in_A) * ln_10
        ) / (ln_10 * (TO_D(in_M) + TO_D(in_A)));

        in += in_step;
        in_T += in_T_step;
        in_M += in_M_step;
        in_A += in_A_step;
        out += out_step;
    }
}
PyUFuncGenericFunction fasinh_func[1] = {&double_fasinh};
static char fasinh_types[5] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static void double_logicle(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    npy_intp n = dimensions[0];
    char *in = args[0], *in_T = args[1], *in_W = args[2], *in_M = args[3], *in_A = args[4], *in_tol = args[5];
    char *out = args[6];
    npy_intp in_step = steps[0], in_T_step = steps[1], in_W_step = steps[2], in_M_step = steps[3], in_A_step = steps[4], in_tol_step = steps[5];
    npy_intp out_step = steps[6];

    for (npy_intp i = 0; i < n; ++i) {
        TO_D(out) = logicle(TO_D(in), TO_D(in_T), TO_D(in_W), TO_D(in_M), TO_D(in_A), TO_D(in_tol));

        in += in_step;
        in_T += in_T_step;
        in_W += in_W_step;
        in_M += in_M_step;
        in_A += in_A_step;
        in_tol += in_tol_step;
        out += out_step;
    }
}
PyUFuncGenericFunction logicle_func[1] = {&double_logicle};
static char logicle_types[7] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static void double_hyperlog(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    npy_intp n = dimensions[0];
    char *in = args[0], *in_T = args[1], *in_W = args[2], *in_M = args[3], *in_A = args[4], *in_tol = args[5];
    char *out = args[6];
    npy_intp in_step = steps[0], in_T_step = steps[1], in_W_step = steps[2], in_M_step = steps[3], in_A_step = steps[4], in_tol_step = steps[5];
    npy_intp out_step = steps[6];

    for (npy_intp i = 0; i < n; ++i) {
        TO_D(out) = hyperlog(TO_D(in), TO_D(in_T), TO_D(in_W), TO_D(in_M), TO_D(in_A), TO_D(in_tol));

        in += in_step;
        in_T += in_T_step;
        in_W += in_W_step;
        in_M += in_M_step;
        in_A += in_A_step;
        in_tol += in_tol_step;
        out += out_step;
    }
}
PyUFuncGenericFunction hyperlog_func[1] = {&double_hyperlog};
static char hyperlog_types[7] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static PyMethodDef FCSMethods[] = {
    {"loadFCS", loadFCS, METH_VARARGS, "Loads an FCS file"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef libfcsmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_libfcs_ext",
    .m_doc = "Extension module for using libfcs to load FCS Python objects",
    .m_size = -1,
    .m_methods = FCSMethods
};

PyMODINIT_FUNC
PyInit__libfcs_ext(void)
{
    if (libfcs_init()) {
        PyObject *module;

        if (PyType_Ready(&FCSType) < 0) {
            return NULL;
        }
        
        module = PyModule_Create(&libfcsmodule);
        if (module == NULL) {
            return NULL;
        }

        // Init the numpy functions
        import_array();
        import_umath();

        // Init the FCS type
        Py_INCREF(&FCSType);
        if (PyModule_AddObject(module, "FCS", (PyObject*) &FCSType) < 0) {
            Py_DECREF(&FCSType);
            Py_DECREF(module);
            return NULL;
        }

        // Init the Numpy ufuncs
        PyObject *d = PyModule_GetDict(module);
        // tunable linear
        PyObject *flin = PyUFunc_FromFuncAndData(flin_func, NULL, flin_types, 1, 3, 1,
                                                 PyUFunc_None, "flin",
                                                 "flin_docstring", 0);
        PyDict_SetItemString(d, "flin", flin);
        Py_DECREF(flin);
        // tunable log
        PyObject *flog = PyUFunc_FromFuncAndData(flog_func, NULL, flog_types, 1, 3, 1,
                                                 PyUFunc_None, "flog",
                                                 "flog_docstring", 0);
        PyDict_SetItemString(d, "flog", flog);
        Py_DECREF(flog);

        // tunable asinh
        PyObject *fasinh = PyUFunc_FromFuncAndData(fasinh_func, NULL, fasinh_types, 1, 4, 1,
                                                 PyUFunc_None, "fasinh",
                                                 "fasinh_docstring", 0);
        PyDict_SetItemString(d, "fasinh", fasinh);
        Py_DECREF(fasinh);

        // logicle
        PyObject *logicle = PyUFunc_FromFuncAndData(logicle_func, NULL, logicle_types, 1, 6, 1,
                                                 PyUFunc_None, "logicle",
                                                 "logicle_docstring", 0);
        PyDict_SetItemString(d, "logicle", logicle);
        Py_DECREF(logicle);

        // logicle
        PyObject *hyperlog = PyUFunc_FromFuncAndData(hyperlog_func, NULL, hyperlog_types, 1, 6, 1,
                                                 PyUFunc_None, "hyperlog",
                                                 "hyperlog_docstring", 0);
        PyDict_SetItemString(d, "hyperlog", hyperlog);
        Py_DECREF(hyperlog);

        return module;
    }
    return NULL;
}