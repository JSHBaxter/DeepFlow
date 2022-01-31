#ifndef HMF_MEANPASS1D_CPU_SOLVER_H
#define HMF_MEANPASS1D_CPU_SOLVER_H

#include "hmf_trees.h"
#include "hmf_meanpass_cpu_solver.h"

class HMF_MEANPASS_CPU_SOLVER_1D : public HMF_MEANPASS_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const float* rx;
    const float* rx_b;

protected:
    int min_iter_calc();
    void init_reg_info();
    void clean_up();
    void update_spatial_flow_calc();
    void parity_mask_buffer(float* buffer, const int parity);
    void parity_merge_buffer(float* buffer, const float* other, const int parity);
    
public:
    ~HMF_MEANPASS_CPU_SOLVER_1D();
    HMF_MEANPASS_CPU_SOLVER_1D(
        const bool channels_first,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_c,
        const int n_r,
        const int sizes[1],
        const float* data_cost,
        const float* rx_cost,
		const float* init_u,
        float* u );
};

class HMF_MEANPASS_CPU_GRADIENT_1D : public HMF_MEANPASS_CPU_GRADIENT_BASE
{
private:
    const int n_x;
    float* g_rx;
	const float* rx;
    const float* rx_b;

protected:
    int min_iter_calc();
    void init_reg_info();
    void clean_up();
    void get_reg_gradients_and_push(float tau);
    
public:
    ~HMF_MEANPASS_CPU_GRADIENT_1D();
    HMF_MEANPASS_CPU_GRADIENT_1D(
        const bool channels_first,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_c,
        const int n_r,
        const int sizes[1],
        const float* u,
        const float* g,
		const float* rx_cost,
        float* g_d,
        float* g_rx );
};

#endif
