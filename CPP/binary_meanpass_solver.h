    
#ifndef BINARY_MEANPASS_SOLVER_H
#define BINARY_MEANPASS_SOLVER_H

#include "algorithm.h"
#include "spatial_meanpass.h"

template<typename DEV>
class BINARY_MEANPASS_SOLVER : public MAXFLOW_ALGORITHM<DEV>
{
private:

protected:
    SPATIAL_MEANPASS_FORWARD_SOLVER<DEV>* spatial_flow;
    const int n_c;
    const int n_s;
    const float * const data;
    const float * const init_u;
    float* const u;
    float* r_eff;
    
    // optimization constants
	const float beta = 0.001f;
	const float epsilon = 0.0001f;
	const float tau = 0.5f;
    
    void block_iter(int parity);
    
public:
    BINARY_MEANPASS_SOLVER(
        const DEV & dev,
        const int dim,
        const int* dims,
        const int n_c,
        const float* const* const inputs,
        float* const u);
        
    ~BINARY_MEANPASS_SOLVER();
    
    void allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over);
    int get_buffer_size();
    
    void run();
};

template class BINARY_MEANPASS_SOLVER<CPU_DEVICE>;
#ifdef USE_CUDA
template class BINARY_MEANPASS_SOLVER<CUDA_DEVICE>;
#endif


template<typename DEV>
class BINARY_MEANPASS_GRADIENT : public MAXFLOW_ALGORITHM<DEV>
{
private:

protected:
    SPATIAL_MEANPASS_BACKWARD_SOLVER<DEV>* spatial_flow;
    const int n_c;
    const int n_s;
    const float * const logits;
    const float * const grad;
    float* const g_data;
    float* u;
	float* d_y;
	float* g_u;
    
    // optimization constants
	const float beta = 0.0001f;
	const float epsilon = 0.01f;
	const float tau = 0.25f;
    
    void block_iter();
    
public:
    BINARY_MEANPASS_GRADIENT(
        const DEV & dev,
        const int dim,
        const int* dims,
        const int n_c,
        const float *const *const inputs,
        float *const *const outputs);
        
    ~BINARY_MEANPASS_GRADIENT();
    
    void allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over);
    int get_buffer_size();
    
    void run();
};

template class BINARY_MEANPASS_GRADIENT<CPU_DEVICE>;
#ifdef USE_CUDA
template class BINARY_MEANPASS_GRADIENT<CUDA_DEVICE>;
#endif

#endif //BINARY_MEANPASS_SOLVER_H
