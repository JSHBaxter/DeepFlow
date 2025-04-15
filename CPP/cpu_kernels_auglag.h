/*

This is a collection of the CPU kernels specific to the Augmented Lagrangian max a posteriori solver.
This is to make these functions easier to find and give a bit more flexibility with naming.

Note that some functions may have multiple signatures. This is for handling data of different dimensionality.
Many functions operate point-wise and thus only one signature is needed regardless of image dimensionality.

*/

#ifndef CPU_KERNELS_AUGLAG
#define CPU_KERNELS_AUGLAG

#include "algorithm.h"

//Functions speficially for augmented lagrangian
void compute_source_flow(const CPU_DEVICE & dev, const float* u, float* ps, const float* pt, const float* div, const float icc, const int n_c, const int n_s);
void compute_sink_flow(const CPU_DEVICE & dev, const float* u, const float* ps, float* pt, const float* div, const float* d, const float icc, const int n_c, const int n_s);
void compute_multipliers(const CPU_DEVICE & dev, float* erru, float* u, const float* ps, const float* pt, const float* div, const float cc, const int n_c, const int n_s);
void update_source_sink_multiplier_potts(const CPU_DEVICE& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_c, const int n_s);
void update_source_sink_multiplier_binary(const CPU_DEVICE& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_s);
void calc_capacity_potts(const CPU_DEVICE & dev,float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau);
void calc_capacity_binary(const CPU_DEVICE & dev,float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const float icc, const float tau);
void update_spatial_flows(const CPU_DEVICE& dev, const float* const g, float* const div, float *const *const p, const float *const *const r, const int dim, const int* const n, const int n_c);
void update_spatial_star_flows(const CPU_DEVICE& dev, const float* const g, float* const div, float *const *const p, const float *const *const r, const float *const *const l, const int dim, const int* const n, const int n_c);
void init_flows_binary(const CPU_DEVICE & dev,const float* d, float* ps, float* pt, float* u, const int n_s);
void init_flows_potts(const CPU_DEVICE & dev,const float* d, float* ps, float* pt, float* u, const int n_s, const int n_c);
void init_flows(const CPU_DEVICE & dev,const float* d, float* ps, const int n_c, const int n_s);


#endif