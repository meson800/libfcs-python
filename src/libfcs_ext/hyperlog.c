// Implementation inspired by https://github.com/RGLab/flowCore/blob/master/src/Hyperlog.cpp
// which is licensed under the Artistic 2.0 license
// by B Ellis, P Haaland, F Hahne, N Le Meur, N Gopalakrishnan, J Spidlen, M Jiang, G Finak
// and S Granjeaud
// The implementation has been chiefly modified with a simplified parameter cache
#include <math.h>
#include <stdbool.h>

#include "hyperlog.h"

#define TAYLOR_LENGTH 16

struct HyperlogParams {
    double T;
    double W;
    double M;
    double A;
    double a;
    double b;
    double c;
    double f;
    double w;
    double x1;
    double inverse_x0;
    double taylor_cutoff;
    double taylor[TAYLOR_LENGTH];
};

#define CACHE_SIZE 5
static struct HyperlogParams hyperlog_coeff_cache[CACHE_SIZE] = {{0}};
static int next_cache = 0;

//TODO: have the user malloc the cache so we are thread-safe
double inverse_hyperlog_param(double y, struct HyperlogParams* params);

void generate_hyperlog_params(double T, double W, double M, double A, int slot) {
    // Generate parameters into a given slot
    struct HyperlogParams* params = hyperlog_coeff_cache + slot;
    params->T = T;
    params->W = W;
    params->M = M;
    params->A = A;

    double w = W / (M + A);
    double x2 = A / (M + A);
    double x1 = x2 + w;
    double x0 = x2 + 2 * w;
    double b = (M + A) * log(10.0);

    double e2bx0 = exp(b * x0);
    double c_a = e2bx0 / w;
    double f_a = exp(b * x1) + c_a * x1;

    double a = T / ((exp(b) + c_a) - f_a);
    double c = c_a * a;
    double f = f_a * a;

    params->a = a;
    params->b = b;
    params->c = c;
    params->f = f;
    params->w = w;
    params->x1 = x1;

    params->taylor_cutoff = x1 + w / 4.0;
    // Compute taylor series
    double coeff = a * exp(b * x1);
    for (int i = 0; i < TAYLOR_LENGTH; ++i) {
        coeff *= b / (i + 1);
        params->taylor[i] = coeff;
    }
    // Implied by the hyperlog condition
    params->taylor[0] += c;
    params->inverse_x0 = inverse_hyperlog_param(x0, params);
}

struct HyperlogParams* lookup_params(struct HyperlogParams* cache, int cache_len,
                                            double T, double W, double M, double A) {
    // Lookup the T/W/M/A info in the cache table.
    int coeff_slot = -1;
    for (int i = 0; i < cache_len; ++i) {
        // Start at the last-filled slot
        int slot = (i + (cache_len - 1) + next_cache) % cache_len;
        if (T == hyperlog_coeff_cache[slot].T && W == hyperlog_coeff_cache[slot].W
            && M == hyperlog_coeff_cache[slot].M && A == hyperlog_coeff_cache[slot].A) {
            coeff_slot = slot;
            break;
        }
    }
    // Generate it if it doesn't exist
    if (coeff_slot == -1) {
        generate_hyperlog_params(T,W,M,A,next_cache);
        coeff_slot = next_cache;
        next_cache = (next_cache + 1) % cache_len;
    }
    return hyperlog_coeff_cache + coeff_slot;
}

double inverse_hyperlog_param(double y, struct HyperlogParams* params) {
    // Reflect negative values
    bool is_negative = y < params->x1;
    if (is_negative) {
        y = 2 * params->x1 - y;
    }

    double x;
    // Is this really a correct inverse in the Taylor cutoff region?
    if (y < params->taylor_cutoff) {
        double recentered_y = y - params->x1;
        x = 0.0;
        for (int i = TAYLOR_LENGTH - 1; i >= 0; --i) {
            x = (x + params->taylor[i]) * recentered_y;
        }
    } else {
        x = (params->a * exp(params->b * y) + params->c * y) - params->f;
    }

    if (is_negative) {
        return -x;
    }
    return x;
}

double hyperlog_param(double val, double tol, struct HyperlogParams* params) {
    // Solve!
    // Easy case first: if x = 0, the answer is x1
    if (val == 0) {
        return params->x1;
    }
    // Reflect IC if needed, then reflect back at end
    bool is_negative = val < 0;
    if (is_negative) {
        val = -val;
    }
    // Initial guess is either linear or log, depending on regime
    double x;
    if (val < params->inverse_x0) {
        x = params->x1 + val * params->w / params->inverse_x0;
    } else {
        x = log(val / params->a) / params->b;
    }
    for (int i = 0; i < 20; ++i) {
        double ae2bx = params->a * exp(params->b * x);
        double y;
        if (x < params->taylor_cutoff) {
            double recentered_x = x - params->x1;
            y = 0.0;
            for (int i = TAYLOR_LENGTH - 1; i >= 0; --i) {
                y = (y + params->taylor[i]) * recentered_x;
            }
            y -= val;
        } else {
            y = (ae2bx + params->c * x) - (params->f + val);
        }

        // Compute derivatives for Halley's method
        double abe2bx = params->b * ae2bx;
        double dy = abe2bx + params->c;
        double ddy = params->b * abe2bx;

        double delta = y / (dy * (1 - y * ddy / (2 * dy * dy)));
        x -= delta;

        if (fabs(delta) < tol) {
            if (is_negative) {
                return 2 * params->x1 - x;
            } else {
                return x;
            }
        }
    }
    // TODO: Add exception? Or NaN?
    return nan("");
}

double hyperlog(double val, double T, double W, double M, double A, double tol) {
    return hyperlog_param(val,tol,lookup_params(hyperlog_coeff_cache, CACHE_SIZE, T, W, M, A));
}

double inverse_hyperlog(double val, double T, double W, double M, double A) {
    return inverse_hyperlog_param(val,lookup_params(hyperlog_coeff_cache, CACHE_SIZE, T, W, M, A));
}