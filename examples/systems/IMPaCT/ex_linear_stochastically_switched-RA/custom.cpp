#include "../../src/IMDP.h"

/// Custom PDF function, change this to the PDF function desired that will be integrated over with Monte Carlo integration
double customPDF(double *x, size_t dim, void *params)
{
    //custom PDF parameters that are passed in (not all need to be used)
    customParams *p = reinterpret_cast<customParams*>(params);
    vec mean = p->mean;
    vec state_start = p->state_start;
    //function<vec(const vec&)> dynamics1 = p->dynamics1;
    vec input = p->input;
    vec lb = p->lb;
    vec ub = p->ub;
    vec eta = p->eta;

    // Stochastically switched system with additive Gaussian noise for each mode
    mat A1 = {{0.1, 0.9}, {0.8, 0.2}};
    vec sigma1 = {0.3, 0.2};
    vec mode1_mean = A1 * mean;

    mat A2 = {{0.8, 0.2}, {0.1, 0.9}};
    vec sigma2 = {0.2, 0.1};
    vec mode2_mean = A2 * mean;

    vec bernoulli_selection = {0.7, 0.3};

    double mode1_pdf = 1.0;
    for (size_t i = 0; i < dim; ++i) {
        mode1_pdf *= exp(-0.5 * pow(x[i] - mode1_mean[i], 2.0) / pow(sigma1[i], 2.0)) / sqrt(2 * M_PI * pow(sigma1[i], 2.0));
    }

    double mode2_pdf = 1.0;
    for (size_t i = 0; i < dim; ++i) {
        mode2_pdf *= exp(-0.5 * pow(x[i] - mode2_mean[i], 2.0) / pow(sigma2[i], 2.0)) / sqrt(2 * M_PI * pow(sigma2[i], 2.0));
    }

    return bernoulli_selection[0] * mode1_pdf + bernoulli_selection[1] * mode2_pdf;
}