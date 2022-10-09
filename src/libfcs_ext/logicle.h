#ifndef _LIBFCS_EXT_LOGICLE
#define _LIBFCS_EXT_LOGICLE

struct LogicleParamCache;

struct LogicleParamCache* init_logicle_cache();
void free_logicle_cache(struct LogicleParamCache* params);

double logicle(double x, double T, double W, double M, double A, double tol, struct LogicleParamCache* cache);
double inverse_logicle(double x, double T, double W, double M, double A, struct LogicleParamCache* cache);

#endif