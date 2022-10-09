#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_13_API_VERSION
#include <math.h>
#include <assert.h>

#include <Python.h>
#include "numpy/npy_common.h"
#include "numpy/ndarrayobject.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#include <fcs.h>
#include "logicle.h"
#include "hyperlog.h"

// Overall helper functions
static PyObject* get_optional_string(OptionalString* ostr, const char* error_msg) {
    if (ostr->present) {
        PyObject* str = PyUnicode_DecodeUTF8(
            (char*)ostr->string.buffer, ostr->string.length, "strict"
        );
        if (str == NULL) {
            PyErr_SetString(PyExc_ValueError, error_msg);
        }
        return str;
    }
    Py_RETURN_NONE;
}

static PyObject* get_optional_float(OptionalFloat* oflt, const char* error_msg) {
    if (oflt->present){
        PyObject* flt = PyFloat_FromDouble(oflt->value);
        if (flt == NULL) {
            PyErr_SetString(PyExc_ValueError, error_msg);
        }
        return flt;
    }
    Py_RETURN_NONE;
}

static PyObject* get_optional_int(OptionalInt64* oint, const char* error_msg) {
    if (oint->present){
        PyObject* p_int = PyLong_FromLongLong(oint->value);
        if (p_int == NULL) {
            PyErr_SetString(PyExc_ValueError, error_msg);
        }
        return p_int;
    }
    Py_RETURN_NONE;
}

// Forward declarations needed
typedef struct FCSObject FCSObject;
// Main object structs
typedef struct {
    PyObject_HEAD
    /* Type-specific fields */
    FCSObject* parent;
    Py_ssize_t param_idx;
} FCSParameter;

typedef struct {
    PyObject_HEAD
    /* Type-specific fields */
    FCSObject* parent;
} FCSParameterList;

typedef struct FCSObject {
    PyObject_HEAD
    /* Type-specific fields */
    FCSFile* file;
    FCSParameterList* event_params;
} FCSObject;


static void FCSParameter_dealloc(FCSParameter *self) {
    Py_DECREF((PyObject*)self->parent);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject* FCSParameter_get_short_name(FCSParameter *self, void *_closure) {
    StringUTF8* str = &self->parent->file->metadata.parameters[self->param_idx].short_name;
    PyObject* p_str = PyUnicode_DecodeUTF8(
        (char*)str->buffer, str->length, "strict"
    );
    if (p_str == NULL) {
        PyErr_SetString(PyExc_ValueError, "Unable to decode parameter short name");
    }
    return p_str;
}

static PyObject* FCSParameter_get_name(FCSParameter *self, void *_closure) {
    return get_optional_string(
        &self->parent->file->metadata.parameters[self->param_idx].name,
        "Unable to decode parameter name"
    );
}

static PyGetSetDef FCSParameter_getsetters[] = {
        {"name", (getter) FCSParameter_get_name, NULL,
            "Full channel name", NULL},
        {"short_name", (getter) FCSParameter_get_short_name, NULL,
            "Short channel name", NULL},
    {NULL} /* Sentinel */
};

static PyTypeObject FCSParameter_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_libfcs_ext.Parameter",
    .tp_doc = PyDoc_STR("FCS parameter"),
    .tp_basicsize = sizeof(FCSParameter),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_FINALIZE,
    .tp_dealloc = FCSParameter_dealloc,
    .tp_getset = FCSParameter_getsetters,
};

static PyObject * FCSParameter_factory(FCSObject* parent, Py_ssize_t param_idx) {
    FCSParameter *self;
    self = (FCSParameter*) FCSParameter_Type.tp_alloc(&FCSParameter_Type, 0);
    if (self != NULL) {
        Py_INCREF(parent);
        self->parent = parent;
        self->param_idx = param_idx;
    }
    return (PyObject *) self;
}


static Py_ssize_t FCSParameterListLength(PyObject *self) {
    FCSParameterList* object = (FCSParameterList*) self;
    return object->parent->file->metadata.n_parameters;
}

static PyObject* FCSParameterListGetItem(PyObject *self, Py_ssize_t i) {
    FCSParameterList* object = (FCSParameterList*) self;
    // Bounds check
    if (i >= object->parent->file->metadata.n_parameters) {
        return NULL;
    }
    return FCSParameter_factory(object->parent, i);
}

static PySequenceMethods FCSParameterListSeqMethods = {
    .sq_length = FCSParameterListLength,
    .sq_item = FCSParameterListGetItem,
};

static int FCSParameterList_traverse(FCSParameterList* self, visitproc visit, void *arg) {
    Py_VISIT(self->parent);
    return 0;
}

static  void FCSParameterList_dealloc(FCSParameterList *self) {
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->parent);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyTypeObject FCSParameterList_Type = {
    PyVarObject_HEAD_INIT(NULL,0)
    .tp_name = "_libfcs_ext.ParameterList",
    .tp_doc = PyDoc_STR("FCS Parameter list"),
    .tp_basicsize = sizeof(FCSParameterList),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_new = NULL,
    .tp_dealloc = FCSParameterList_dealloc,
    .tp_traverse = FCSParameterList_traverse,
    .tp_as_sequence = &FCSParameterListSeqMethods,
};


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

        puts("About to create a ParameterList");
        self->event_params = (FCSParameterList*)FCSParameterList_Type.tp_alloc(&FCSParameterList_Type, 0);
        puts("finished calling tp_alloc");
        if (self->event_params != NULL) {
            Py_INCREF(self);
            self->event_params->parent = self;
            puts("Created parameter list");
        }
    }
    return (PyObject *) self;
}

static void FCSObject_dealloc(FCSObject *self) {
    free_FCS(self->file);
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->event_params);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int FCSObject_traverse(FCSObject* self, visitproc visit, void *arg) {
    Py_VISIT(self->event_params);
    return 0;
}

static PyObject* FCSObject_get_n_events_aborted(FCSObject *self, void *_closure) {
    if (self->file->metadata.n_events_aborted.present) {
        return Py_BuildValue("L", self->file->metadata.n_events_aborted.value);
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
    return get_optional_string(&self->file->metadata.cytometer_type, "Unable to decode cytometer type");
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
    PyArrayObject* np_array = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimensions, NPY_DOUBLE, self->file->uncompensated.data);
    if (np_array == NULL) {
        return NULL;
    }
    Py_INCREF(self);
    if (PyArray_SetBaseObject(np_array, (PyObject*)self) != 0) {
        Py_DECREF(self);
        Py_DECREF(np_array);
        return NULL;
    }
    return (PyObject*)np_array;
}

static PyObject* FCSObject_get_parameters(FCSObject* self, void *_closure) {
    Py_INCREF(self->event_params);
    return (PyObject*)self->event_params;
}

static PyObject* FCSObject_get_compensated(FCSObject *self, void *_closure) {
    npy_intp dimensions[2] = {self->file->compensated.n_rows, self->file->compensated.n_cols};
    PyArrayObject* np_array = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimensions, NPY_DOUBLE, self->file->compensated.data);
    if (np_array == NULL) {
        return NULL;
    }
    Py_INCREF(self);
    if (PyArray_SetBaseObject(np_array, (PyObject*)self) != 0) {
        Py_DECREF(self);
        Py_DECREF(np_array);
        return NULL;
    }
    return (PyObject*)np_array;
}

static PyGetSetDef FCSObject_getsetters[] = {
        {"parameters", (getter) FCSObject_get_parameters, NULL,
            "Event parameters", NULL},
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
        {"acquire_date_str", (getter) FCSObject_get_acquire_date, NULL,
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
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_FINALIZE | Py_TPFLAGS_HAVE_GC,
    .tp_new = FCSObject_new,
    .tp_dealloc = FCSObject_dealloc,
    .tp_traverse = FCSObject_traverse,
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

    /*
    printf("Loaded FCS file with %zd parameters and %zd events\n", fcs->compensated.n_cols, fcs->compensated.n_rows);
    if (fcs->metadata.comment.present) {
        printf("\t%.*s\n",
                fcs->metadata.comment.string.length,
                fcs->metadata.comment.string.buffer
        );
    }
    printf("\tmode=%Id\n\tdatatype=%Id\n", fcs->metadata.mode, fcs->metadata.datatype);
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

    printf("\n\t%Id extra keyvals:\n", fcs->metadata.extra_keyvals.n_vals);
    for (int i = 0; i < fcs->metadata.extra_keyvals.n_vals; ++i) {
        printf("\t\t%.*s: %.*s\n",
            fcs->metadata.extra_keyvals.items[i].key.length,
            fcs->metadata.extra_keyvals.items[i].key.buffer,
            fcs->metadata.extra_keyvals.items[i].value.length,
            fcs->metadata.extra_keyvals.items[i].value.buffer
        );
    }

    */
    Py_DECREF(filename_bytes);


    Py_RETURN_NONE;
}

/************** ufuncs for fast Numpy processing *******************/
#define TO_D(x) (*((double*)(x)))
#define TO_BOOL(x) (*((npy_bool*)(x)))
static void *null_data[1] = {NULL};
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

static void double_inv_flin(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    npy_intp n = dimensions[0];
    char *in = args[0], *in_T = args[1], *in_A = args[2];
    char *out = args[3];
    npy_intp in_step = steps[0], in_T_step = steps[1], in_A_step = steps[2];
    npy_intp out_step = steps[3];

    for (npy_intp i = 0; i < n; ++i) {
        TO_D(out) = TO_D(in) * (TO_D(in_T) + TO_D(in_A)) - TO_D(in_A);

        in += in_step;
        in_T += in_T_step;
        in_A += in_A_step;
        out += out_step;
    }
}
PyUFuncGenericFunction inv_flin_func[1] = {&double_inv_flin};
static char inv_flin_types[4] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

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

static void double_inv_flog(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    npy_intp n = dimensions[0];
    char *in = args[0], *in_T = args[1], *in_M = args[2];
    char *out = args[3];
    npy_intp in_step = steps[0], in_T_step = steps[1], in_M_step = steps[2];
    npy_intp out_step = steps[3];

    for (npy_intp i = 0; i < n; ++i) {
        TO_D(out) = pow(10.0, (TO_D(in) - 1.0) * TO_D(in_M)) * TO_D(in_T);

        in += in_step;
        in_T += in_T_step;
        in_M += in_M_step;
        out += out_step;
    }
}
PyUFuncGenericFunction inv_flog_func[1] = {&double_inv_flog};
static char inv_flog_types[4] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

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

static void double_inv_fasinh(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    npy_intp n = dimensions[0];
    char *in = args[0], *in_T = args[1], *in_M = args[2], *in_A = args[3];
    char *out = args[4];
    npy_intp in_step = steps[0], in_T_step = steps[1], in_M_step = steps[2], in_A_step = steps[3];
    npy_intp out_step = steps[4];

    double ln_10 = log(10.0);

    for (npy_intp i = 0; i < n; ++i) {
        TO_D(out) = (
            TO_D(in_T) * sinh(
                            ln_10 * (-TO_D(in_A) + (TO_D(in_A) + TO_D(in_M)) * TO_D(in))
            ) / sinh(ln_10 * TO_D(in_M))
        );

        in += in_step;
        in_T += in_T_step;
        in_M += in_M_step;
        in_A += in_A_step;
        out += out_step;
    }
}
PyUFuncGenericFunction inv_fasinh_func[1] = {&double_inv_fasinh};
static char inv_fasinh_types[5] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static void double_logicle(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    npy_intp n = dimensions[0];
    char *in = args[0], *in_T = args[1], *in_W = args[2], *in_M = args[3], *in_A = args[4], *in_tol = args[5];
    char *out = args[6];
    npy_intp in_step = steps[0], in_T_step = steps[1], in_W_step = steps[2], in_M_step = steps[3], in_A_step = steps[4], in_tol_step = steps[5];
    npy_intp out_step = steps[6];

    struct LogicleParamCache* cache = init_logicle_cache();

    for (npy_intp i = 0; i < n; ++i) {
        TO_D(out) = logicle(TO_D(in), TO_D(in_T), TO_D(in_W), TO_D(in_M), TO_D(in_A), TO_D(in_tol), cache);

        in += in_step;
        in_T += in_T_step;
        in_W += in_W_step;
        in_M += in_M_step;
        in_A += in_A_step;
        in_tol += in_tol_step;
        out += out_step;
    }
    free_logicle_cache(cache);
}
PyUFuncGenericFunction logicle_func[1] = {&double_logicle};
static char logicle_types[7] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static void double_inv_logicle(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    npy_intp n = dimensions[0];
    char *in = args[0], *in_T = args[1], *in_W = args[2], *in_M = args[3], *in_A = args[4];
    char *out = args[5];
    npy_intp in_step = steps[0], in_T_step = steps[1], in_W_step = steps[2], in_M_step = steps[3], in_A_step = steps[4];
    npy_intp out_step = steps[5];

    struct LogicleParamCache* cache = init_logicle_cache();

    for (npy_intp i = 0; i < n; ++i) {
        TO_D(out) = inverse_logicle(TO_D(in), TO_D(in_T), TO_D(in_W), TO_D(in_M), TO_D(in_A), cache);

        in += in_step;
        in_T += in_T_step;
        in_W += in_W_step;
        in_M += in_M_step;
        in_A += in_A_step;
        out += out_step;
    }
    free_logicle_cache(cache);
}
PyUFuncGenericFunction inv_logicle_func[1] = {&double_inv_logicle};
static char inv_logicle_types[6] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static void double_hyperlog(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    npy_intp n = dimensions[0];
    char *in = args[0], *in_T = args[1], *in_W = args[2], *in_M = args[3], *in_A = args[4], *in_tol = args[5];
    char *out = args[6];
    npy_intp in_step = steps[0], in_T_step = steps[1], in_W_step = steps[2], in_M_step = steps[3], in_A_step = steps[4], in_tol_step = steps[5];
    npy_intp out_step = steps[6];

    struct HyperlogParamCache* cache = init_hyperlog_cache();

    for (npy_intp i = 0; i < n; ++i) {
        TO_D(out) = hyperlog(TO_D(in), TO_D(in_T), TO_D(in_W), TO_D(in_M), TO_D(in_A), TO_D(in_tol), cache);

        in += in_step;
        in_T += in_T_step;
        in_W += in_W_step;
        in_M += in_M_step;
        in_A += in_A_step;
        in_tol += in_tol_step;
        out += out_step;
    }
    free_hyperlog_cache(cache);
}
PyUFuncGenericFunction hyperlog_func[1] = {&double_hyperlog};
static char hyperlog_types[7] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static void double_inv_hyperlog(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    npy_intp n = dimensions[0];
    char *in = args[0], *in_T = args[1], *in_W = args[2], *in_M = args[3], *in_A = args[4];
    char *out = args[5];
    npy_intp in_step = steps[0], in_T_step = steps[1], in_W_step = steps[2], in_M_step = steps[3], in_A_step = steps[4];
    npy_intp out_step = steps[5];

    struct HyperlogParamCache* cache = init_hyperlog_cache();

    for (npy_intp i = 0; i < n; ++i) {
        TO_D(out) = inverse_hyperlog(TO_D(in), TO_D(in_T), TO_D(in_W), TO_D(in_M), TO_D(in_A), cache);

        in += in_step;
        in_T += in_T_step;
        in_W += in_W_step;
        in_M += in_M_step;
        in_A += in_A_step;
        out += out_step;
    }
    free_hyperlog_cache(cache);
}
PyUFuncGenericFunction inv_hyperlog_func[1] = {&double_inv_hyperlog};
static char inv_hyperlog_types[6] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static PyMethodDef FCSMethods[] = {
    {"loadFCS", loadFCS, METH_VARARGS, "Loads an FCS file"},
    {NULL, NULL, 0, NULL}
};

static void double_polygon(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data)
{
    // To understand the madness, use
    // https://github.com/numpy/numpy/blob/aeb39dfb3566e853a09082bfe9438ef118916be7/numpy/core/src/umath/matmul.c.src
    //
    // Our signature here is (n,2),(m,2)->(n)
    // for `n` points with `m` polygon vertices
    npy_intp outer = dimensions[0];
    npy_intp n = dimensions[1];
    assert(dimensions[2] == 2);
    npy_intp m = dimensions[3];
    assert(dimensions[4] == 2);
    // Steps to take for each outer loop
    npy_intp outer_event_step = steps[0], outer_vertex_step = steps[1], outer_output_step = steps[2];
    // Steps for the events
    npy_intp event_step_n = steps[3], event_step_2 = steps[4];
    npy_intp vertex_step_m = steps[5], vertex_step_2 = steps[6];
    npy_intp output_step_n = steps[7];
    char *outer_event = args[0], *outer_vertex = args[1], *outer_output = args[2];
    printf("outer: %Id, n: %Id, m: %Id\n\toes: %Id, ovs: %Id, oos: %Id\n\tesn: %Id, es2: %Id, vsm: %Id, vs2: %Id, osn: %Id\n",
        outer, n, m, outer_event_step, outer_vertex_step, outer_output_step, event_step_n, event_step_2,
        vertex_step_m, vertex_step_2, output_step_n);
    
    for (npy_intp outer_idx = 0; outer_idx < outer; ++outer_idx,
        outer_event += outer_event_step, outer_vertex += outer_vertex_step,
        outer_output += outer_output_step) {
        char *event = outer_event;
        char *vertex = outer_vertex;
        char *output = outer_output;
        for (npy_intp event_idx = 0; event_idx < n; ++event_idx,
            event += event_step_n, output += output_step_n) {
            // For each event, check if a ray projected straight upward
            // from the event intersects an even or odd number of polygon
            // line segments
            char *first_vertex = vertex;

            TO_BOOL(output) = NPY_FALSE;
            for (npy_intp vertex_idx = 0; vertex_idx < m; ++vertex_idx, first_vertex += vertex_step_m) {
                // Each vertex is associated with a line segment.
                char *second_vertex = first_vertex;
                if (vertex_idx < m - 1) {
                    second_vertex += vertex_step_m;
                } else {
                    // If we are on the last vertex, loop back to the
                    // first vertex to complete the line segment
                    second_vertex = vertex;
                }

                double e_x = TO_D(event), e_y = TO_D(event + event_step_2);
                double x_1 = TO_D(first_vertex), y_1 = TO_D(first_vertex + vertex_step_2);
                double x_2 = TO_D(second_vertex), y_2 = TO_D(second_vertex + vertex_step_2);

                // We only potentially cross if y_1 >= e_y or y_2 >= e_y (e.g. segment is above us)
                // and if x_1 <= e_x < x_2 (for x_1 < x_2)
                // We use this asymmetric condition to symmetry-break the case
                // where we go through a vertex.
                if ((
                        (x_1 < e_x && x_2 >= e_x) ||
                        (x_2 < e_x && x_1 >= e_x)
                    ) && (y_1 >= e_y || y_2 >= e_y)) {
                    // Linearly interpolate to see if this is a true intersection
                    TO_BOOL(output) ^= (y_1 + (e_x - x_1)/(x_2 - x_1) * (y_2 - y_1) > e_y);
                }
            }
        }
    }
}

PyUFuncGenericFunction polygon_gate_func[1] = {&double_polygon};
static char polygon_gate_types[3] = {NPY_DOUBLE, NPY_DOUBLE, NPY_BOOL};
static char* polygon_gate_signature = "(n,2),(m,2)->(n)";

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
    puts("About to init");
    if (libfcs_init()) {
        PyObject *module;
        puts("Haskell inited");

        // Init various types
        if (PyType_Ready(&FCSType) < 0) {
            return NULL;
        }
        if (PyType_Ready(&FCSParameterList_Type) < 0) {
            return NULL;
        }
        if (PyType_Ready(&FCSParameter_Type) < 0) {
            return NULL;
        }
        
        puts("About to create Py module");
        module = PyModule_Create(&libfcsmodule);
        if (module == NULL) {
            return NULL;
        }
        puts("Py module created");

        // Init the numpy functions
        import_array();
        import_ufunc();
        import_umath();
        puts("Numpy inited");

        // Init the various types
        Py_INCREF(&FCSType);
        if (PyModule_AddObject(module, "FCS", (PyObject*) &FCSType) < 0) {
            Py_DECREF(&FCSType);
            Py_DECREF(module);
            return NULL;
        }
        puts("Created the FCS type");

        /* Don't expose the ParameterList and Parameter types; users cannot create them
        Py_INCREF(&FCSParameterList_Type);
        if (PyModule_AddObject(module, "ParameterList", (PyObject*) &FCSParameterList_Type) < 0) {
            Py_DECREF(&FCSParameterList_Type);
            Py_DECREF(module);
            return NULL;
        }
        puts("Created the ParameterList type");

        Py_INCREF(&FCSParameter_Type);
        if (PyModule_AddObject(module, "Parameter", (PyObject*) &FCSParameter_Type) < 0) {
            Py_DECREF(&FCSParameter_Type);
            Py_DECREF(module);
            return NULL;
        }
        puts("Created the Parameter type");
        */



        // Init the Numpy ufuncs
        PyObject *d = PyModule_GetDict(module);
        // tunable linear
        PyObject *flin = PyUFunc_FromFuncAndData(flin_func, null_data, flin_types, 1, 3, 1,
                                                 PyUFunc_None, "flin",
                                                 "flin_docstring", 0);
        PyDict_SetItemString(d, "flin", flin);
        Py_DECREF(flin);
        PyObject *inv_flin = PyUFunc_FromFuncAndData(inv_flin_func, null_data, inv_flin_types, 1, 3, 1,
                                                 PyUFunc_None, "inv_flin",
                                                 "inv_flin_docstring", 0);
        PyDict_SetItemString(d, "inv_flin", inv_flin);
        Py_DECREF(inv_flin);
        puts("Created flin and inv_flin");
        // tunable log
        PyObject *flog = PyUFunc_FromFuncAndData(flog_func, null_data, flog_types, 1, 3, 1,
                                                 PyUFunc_None, "flog",
                                                 "flog_docstring", 0);
        PyDict_SetItemString(d, "flog", flog);
        Py_DECREF(flog);
        PyObject *inv_flog = PyUFunc_FromFuncAndData(inv_flog_func, null_data, inv_flog_types, 1, 3, 1,
                                                 PyUFunc_None, "inv_flog",
                                                 "inv_flog_docstring", 0);
        PyDict_SetItemString(d, "inv_flog", inv_flog);
        Py_DECREF(inv_flog);
        puts("Created flog and inv_flog");

        // tunable asinh
        PyObject *fasinh = PyUFunc_FromFuncAndData(fasinh_func, null_data, fasinh_types, 1, 4, 1,
                                                 PyUFunc_None, "fasinh",
                                                 "fasinh_docstring", 0);
        PyDict_SetItemString(d, "fasinh", fasinh);
        Py_DECREF(fasinh);
        PyObject *inv_fasinh = PyUFunc_FromFuncAndData(inv_fasinh_func, null_data, inv_fasinh_types, 1, 4, 1,
                                                 PyUFunc_None, "inv_fasinh",
                                                 "inv_fasinh_docstring", 0);
        PyDict_SetItemString(d, "inv_fasinh", inv_fasinh);
        Py_DECREF(inv_fasinh);
        puts("Created fasinh and inv_fasinh");

        // logicle
        PyObject *logicle = PyUFunc_FromFuncAndData(logicle_func, null_data, logicle_types, 1, 6, 1,
                                                 PyUFunc_None, "logicle",
                                                 "logicle_docstring", 0);
        PyDict_SetItemString(d, "logicle", logicle);
        Py_DECREF(logicle);
        PyObject *inv_logicle = PyUFunc_FromFuncAndData(inv_logicle_func, null_data, inv_logicle_types, 1, 5, 1,
                                                 PyUFunc_None, "inv_logicle",
                                                 "inv_logicle_docstring", 0);
        PyDict_SetItemString(d, "inv_logicle", inv_logicle);
        Py_DECREF(inv_logicle);
        puts("Created logicle and inv_logicle");

        // logicle
        PyObject *hyperlog = PyUFunc_FromFuncAndData(hyperlog_func, null_data, hyperlog_types, 1, 6, 1,
                                                 PyUFunc_None, "hyperlog",
                                                 "hyperlog_docstring", 0);
        PyDict_SetItemString(d, "hyperlog", hyperlog);
        Py_DECREF(hyperlog);
        PyObject *inv_hyperlog = PyUFunc_FromFuncAndData(inv_hyperlog_func, null_data, hyperlog_types, 1, 5, 1,
                                                 PyUFunc_None, "inv_hyperlog",
                                                 "inv_hyperlog_docstring", 0);
        PyDict_SetItemString(d, "inv_hyperlog", inv_hyperlog);
        Py_DECREF(inv_hyperlog);
        puts("Created hyperlog and inv_hyperlog");

        // polygon gate
        PyObject *polygon_gate = PyUFunc_FromFuncAndDataAndSignature(
            polygon_gate_func, null_data, polygon_gate_types, 1, 2, 1,
            PyUFunc_None, "polygon_gate",
            "polygon_gate_docstring", 0,
            polygon_gate_signature);
        PyDict_SetItemString(d, "polygon_gate", polygon_gate);
        Py_DECREF(polygon_gate);
        puts("Created polygon gate");

        return module;
    }
    return NULL;
}
