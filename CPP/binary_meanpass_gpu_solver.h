    
#ifndef BINARY_MEANPASS_GPU_SOLVER_H
#define BINARY_MEANPASS_GPU_SOLVER_H

#include <cuda_runtime.h>

class BINARY_MEANPASS_GPU_SOLVER_BASE
{
private:

protected:
    const cudaStream_t & dev;
    const int b;
    const int n_c;
    const int n_s;
    const float* const data;
    float* const u;
    float* r_eff;
    
    // optimization constants
	const float beta = 0.001f;
	const float epsilon = 0.0001f;
	const float tau = 0.5f;
    
    virtual int min_iter_calc() = 0;
    virtual void init_vars() = 0;
    virtual void calculate_regularization() = 0;
    virtual void parity_mask_buffer(float* buffer, const int parity) = 0;
    virtual void parity_merge_buffer(float* buffer, const float* other, const int parity) = 0;
    virtual void clean_up() = 0;
    void block_iter(int parity);
    
public:
    BINARY_MEANPASS_GPU_SOLVER_BASE(
        const cudaStream_t & dev,
        const int batch,
        const int n_s,
        const int n_c,
        const float* data_cost,
        const float* init_u,
        float* u,
		float** full_buffs) ;
        
    ~BINARY_MEANPASS_GPU_SOLVER_BASE();
    
    void operator()();
};


class BINARY_MEANPASS_GPU_GRADIENT_BASE
{
private:

protected:
	const cudaStream_t & dev;
    const int b;
    const int n_c;
    const int n_s;
    const float* const logits;
    const float* const grad;
    float* const g_data;
    float* u;
	float* d_y;
	float* g_u;
    
    // optimization constants
	const float beta = 0.0001f;
	const float epsilon = 0.01f;
	const float tau = 0.25f;
    
    virtual int min_iter_calc() = 0;
    virtual void init_vars() = 0;
    virtual void get_reg_gradients_and_push(float tau) = 0;
    virtual void clean_up() = 0;
    void block_iter();
    
public:
    BINARY_MEANPASS_GPU_GRADIENT_BASE(
        const cudaStream_t & dev,
		const int batch,
		const int n_s,
		const int n_c,
		const float* u,
		const float* g,
		float* g_d,
		float** full_buffs);
        
    ~BINARY_MEANPASS_GPU_GRADIENT_BASE();
    
    void operator()();
};

#endif //BINARY_MEANPASS_GPU_SOLVER_H
