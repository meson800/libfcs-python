// Implementation inspired by https://github.com/RGLab/flowCore/blob/master/src/Logicle.cpp
// which is licensed under the Artistic 2.0 license
// by B Ellis, P Haaland, F Hahne, N Le Meur, N Gopalakrishnan, J Spidlen, M Jiang, G Finak
// and S Granjeaud
// The implementation has been chiefly modified with a simplified parameter cache
// and a more efficient method to find d.
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "logicle.h"

#define TAYLOR_LENGTH 16
#define CACHE_SIZE 5

struct LogicleParams {
    double T;
    double W;
    double M;
    double A;
    double a;
    double b;
    double c;
    double d;
    double f;
    double x1;
    double taylor_cutoff;
    double taylor[TAYLOR_LENGTH];
};

struct LogicleParamCache {
    struct LogicleParams params[CACHE_SIZE];
    int next_cache;
};

struct LogicleParamCache* init_logicle_cache() {
    return calloc(sizeof(struct LogicleParamCache), 1);
}

void free_logicle_cache(struct LogicleParamCache* params) {
    free(params);
}

void generate_params(double T, double W, double M, double A, int slot, struct LogicleParamCache* cache) {
    // Generate parameters into a given slot
    struct LogicleParams* params = cache->params + slot;
    params->T = T;
    params->W = W;
    params->M = M;
    params->A = A;

    double w = W / (M + A);
    double x2 = A / (M + A);
    double x1 = x2 + w;
    double x0 = x2 + 2 * w;
    double b = (M + A) * log(10.0);
    // Solve the equation f(d) = 2(ln(d) - ln(b)) + w(d+b) = 0
    // f'(d) = 2/d + w
    // f''(d) = -2/d^2
    // Solve with a simple Newton's method (Halley's method, for cubic convergence)
    double d = b;
    double f_x0 = 1;
    // If w == 0, then d = b is the exact solution
    if (w != 0.0) {
        for (int i = 0; i < 10; ++i) {
            f_x0 = 2 * (log(d) - log(b)) + w * (d + b);

            if (fabs(f_x0) < 1e-12) {
                break;
            }
            // Newton's method:
            // d = d - f_x0 / (2 / d + w);
            // Halley's method (alt formulation: https://en.wikipedia.org/wiki/Halley%27s_method):
            double f_df_ratio = f_x0 / (2/d + w);
            d = d - f_df_ratio / (1 + f_df_ratio / (2 * d + d * d * w));
        }
    }
    double c_a = exp(x0 * (b + d));
    double mf_a = exp(b * x1) - c_a / exp(d * x1);
    double a = T / ((exp(b) - mf_a) - c_a / exp(d));
    double c = c_a * a;
    double f = -mf_a * a;

    params->a = a;
    params->b = b;
    params->c = c;
    params->d = d;
    params->f = f;
    params->x1 = x1;

    params->taylor_cutoff = x1 + w / 4.0;
    // Compute taylor series
    double coeff1 = a * exp(b * x1);
    double coeff2 = -c / exp(d * x1);
    for (int i = 0; i < TAYLOR_LENGTH; ++i) {
        coeff1 *= b / (i + 1);
        coeff2 *= -d / (i + 1);
        params->taylor[i] = coeff1 + coeff2;
    }
    // Implied by the Logicle condition
    params->taylor[1] = 0;
}

static struct LogicleParams* lookup_params(struct LogicleParamCache* cache, int cache_len,
                                            double T, double W, double M, double A) {
    // Lookup the T/W/M/A info in the cache table.
    int coeff_slot = -1;
    for (int i = 0; i < CACHE_SIZE; ++i) {
        // Start at the last-filled slot
        int slot = (i + (CACHE_SIZE - 1) + cache->next_cache) % CACHE_SIZE;
        if (T == cache->params[slot].T && W == cache->params[slot].W
            && M == cache->params[slot].M && A == cache->params[slot].A) {
            coeff_slot = slot;
            break;
        }
    }
    // Generate it if it doesn't exist
    if (coeff_slot == -1) {
        generate_params(T,W,M,A,cache->next_cache, cache);
        coeff_slot = cache->next_cache;
        cache->next_cache = (cache->next_cache + 1) % CACHE_SIZE;
    }
    struct LogicleParams* params = cache->params + coeff_slot;
    return params;
}

static double logicle_params(double val, double tol, struct LogicleParams* params) {
    double T = params->T;
    double W = params->W;
    double M = params->M;
    double A = params->A;
    // Solve!
    // Easy case first: if x = 0, the answer is x1
    if (val == 0) {
        return params->x1;
    }
    // Reflect IC if needed, then reflect back at end
    bool negative = val < 0;
    if (negative) {
        val = -val;
    }
    // Initial guess is either linear or log, depending on regime
    double x;
    if (val < params->f) {
        x = params->x1 + val / params->taylor[0];
    } else {
        x = log(val / params->a) / params->b;
    }
    for (int i = 0; i < 20; ++i) {
        double ae2bx = params->a * exp(params->b * x);
        double ce2mdx = params->c / exp(params->d * x);
        double y;
        if (x < params->taylor_cutoff) {
            // Use the Taylor series
            double centered_x = x - params->x1;
            // Sum from small Taylor coefficients to big ones
            y = 0;
            for (int series_i  = TAYLOR_LENGTH - 1; series_i >= 0; --series_i) {
                y = (y + params->taylor[series_i]) * centered_x;
            }
            // Recenter as needed
            y -= val;
        } else {
            // Use the biexponential expansion
            y = (ae2bx + params->f) - (ce2mdx + val);
        }

        double abe2bx = params->b * ae2bx;
        double cde2mdx = params->d * ce2mdx;
        double dy = abe2bx + cde2mdx;
        double ddy = params->b * abe2bx - params->d * cde2mdx;

        // Halley's method again
        double delta = y / (dy * (1 - y * ddy / (2 * dy * dy)));
        x -= delta;
        if (fabs(delta) < tol) {
            if (negative) {
                return 2 * params->x1 - x;
            } else {
                return x;
            }
        }
    }
    // TODO: Add exception? Or NaN?
    return nan("");
}

static double inverse_logicle_params(double y, struct LogicleParams* params) {
    double T = params->T;
    double W = params->W;
    double M = params->M;
    double A = params->A;

    // Rescale input value as needed
    bool negative = y < params->x1;
    if (negative) {
        y = 2 * params->x1 - y;
    }

    double x;
    if (y < params->taylor_cutoff) {
        // do the series
        double centered_y = y - params->x1;
        // Sum from small Taylor coefficients to big ones
        x = 0;
        for (int series_i  = TAYLOR_LENGTH - 1; series_i >= 0; --series_i) {
            x = (x + params->taylor[series_i]) * centered_y;
        }
    } else {
        x = (params->a * exp(params->b * y) + params->f) - params->c / exp(params->d * y);
    }

    if (negative) {
        return -x;
    }
    return x;
}

double logicle(double val, double T, double W, double M, double A, double tol, struct LogicleParamCache* cache) {
    return logicle_params(val,tol,lookup_params(cache, CACHE_SIZE, T, W, M, A));
}

double inverse_logicle(double val, double T, double W, double M, double A, struct LogicleParamCache* cache) {
    return inverse_logicle_params(val,lookup_params(cache, CACHE_SIZE, T, W, M, A));
}