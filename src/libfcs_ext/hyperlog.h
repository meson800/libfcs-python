#ifndef _LIBFCS_EXT_HYPERLOG
#define _LIBFCS_EXT_HYPERLOG

struct HyperlogParamCache;

struct HyperlogParamCache* init_hyperlog_cache();
void free_hyperlog_cache(struct HyperlogParamCache* params);

double hyperlog(double x, double T, double W, double M, double A, double tol, struct HyperlogParamCache* cache);
double inverse_hyperlog(double y, double T, double W, double M, double A, struct HyperlogParamCache* cache);

#endif