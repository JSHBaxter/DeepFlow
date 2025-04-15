
#ifndef SPATIAL_MEANPASS_SOLVER_H
#define SPATIAL_MEANPASS_SOLVER_H

#include "algorithm.h"

template<typename DEV>
class SPATIAL_MEANPASS_FORWARD_SOLVER : public MAXFLOW_ALGORITHM<DEV>
{
protected:
    const int n_c;
    const int n_s;
    const int* const n;
    const int dim;
    
    const float* u;
    float* r_eff;
    const float *const *const r;

public:
    ~SPATIAL_MEANPASS_FORWARD_SOLVER();
	SPATIAL_MEANPASS_FORWARD_SOLVER(
        const DEV & dev,
        const int n_channels,
        const int* const n,
        const int dim,
        const float* const* const r
	);
    
    int get_min_iter();
    int get_max_iter();
    
    void allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over);
    int get_buffer_size();
    
    void init();
    void parity(float* const buffer, const int n_pc, const int parity);
    void parity(float* const buffer, const float* const other, const int n_pc, const int parity);
    void run();
    void deinit();
};

template class SPATIAL_MEANPASS_FORWARD_SOLVER<CPU_DEVICE>;
#ifdef USE_CUDA
template class SPATIAL_MEANPASS_FORWARD_SOLVER<CUDA_DEVICE>;
#endif

template<typename DEV>
class SPATIAL_MEANPASS_BACKWARD_SOLVER : public MAXFLOW_ALGORITHM<DEV>
{
protected:
    const int n_c;
    const int n_s;
    const int* n;
    const int dim;
    
    float* d_y;
    float* g_u;
    const float* u;
    
    const float *const *const r;
    float *const *const g_r;

public:
    ~SPATIAL_MEANPASS_BACKWARD_SOLVER();
	SPATIAL_MEANPASS_BACKWARD_SOLVER(
        const DEV & dev,
        const int n_channels,
        const int *const n,
        const int dim,
        const float *const *const inputs,
        float *const *const g_r
	);
    
    int get_min_iter();
    int get_max_iter();
    
    void allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over);
    int get_buffer_size();
    
    void init();
    void run();
    void run(float tau);
    void deinit();
};

template class SPATIAL_MEANPASS_BACKWARD_SOLVER<CPU_DEVICE>;
#ifdef USE_CUDA
template class SPATIAL_MEANPASS_BACKWARD_SOLVER<CUDA_DEVICE>;
#endif

#endif