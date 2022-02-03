#ifndef BINARY_MEANPASS1D_CPU_SOLVER_H
#define BINARY_MEANPASS1D_CPU_SOLVER_H

#include "binary_meanpass_cpu_solver.h"
#include "cpu_kernels.h"

class BINARY_MEANPASS_CPU_SOLVER_1D : public BINARY_MEANPASS_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const float * const rx;
	
protected:
    int min_iter_calc();
    void init_vars();
    void calculate_regularization();
    void parity_mask_buffer(float* buffer, const int parity);
    void parity_merge_buffer(float* buffer, const float * const other, const int parity);
    void clean_up();
	
public:
	BINARY_MEANPASS_CPU_SOLVER_1D(
        const bool channels_first,
        const int batch,
        const int n_c,
        const int sizes[1],
        const float * const data_cost,
        const float * const rx_cost,
		const float * const init_u,
        float* u 
	);
};

class BINARY_MEANPASS_CPU_GRADIENT_1D : public BINARY_MEANPASS_CPU_GRADIENT_BASE
{
private:
    const int n_x;
    const float * const rx;
    float* const g_rx;
	
protected:
    int min_iter_calc();
    void init_vars();
	void get_reg_gradients_and_push(float tau);
    void clean_up();

public:
	BINARY_MEANPASS_CPU_GRADIENT_1D(
        const bool channels_first,
        const int batch,
        const int n_c,
        const int sizes[1],
        const float * const u,
        const float * const g,
        const float * const rx_cost,
        float* g_d,
        float* g_rx
	) ;
};

#endif
