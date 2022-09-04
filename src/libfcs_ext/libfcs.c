#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/npy_common.h"
#include "numpy/ndarrayobject.h"
#include "numpy/arrayobject.h"

#include <fcs.h>

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
    puts("About to dealloc FCS file");
    free_FCS(self->file);
    puts("About to dealloc FCS file Python object");
    Py_TYPE(self)->tp_free((PyObject *)self);
    puts("Done deallocing");
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
    printf("About to create numpy array with dims (%d x %d)\n", dimensions[0], dimensions[1]);
    printf("First entry: %f, last entry: %f\n", self->file->uncompensated.data[0], self->file->uncompensated.data[dimensions[0] * dimensions[1] - 1]);
    PyObject* np_array = PyArray_SimpleNewFromData(2, dimensions, NPY_DOUBLE, self->file->uncompensated.data);
    puts("Checking array for null");
    if (np_array == NULL) {
        return NULL;
    }
    puts("About to set PyArray base object");
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

        // Init the FCS type
        Py_INCREF(&FCSType);
        if (PyModule_AddObject(module, "FCS", (PyObject*) &FCSType) < 0) {
            Py_DECREF(&FCSType);
            Py_DECREF(module);
            return NULL;
        }

        return module;
    }
    return NULL;
}