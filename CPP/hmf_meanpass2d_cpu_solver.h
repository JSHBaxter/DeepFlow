#ifndef HMF_MEANPASS2D_CPU_SOLVER_H
#define HMF_MEANPASS2D_CPU_SOLVER_H

#include "hmf_trees.h"
#include "hmf_meanpass_cpu_solver.h"

class HMF_MEANPASS_CPU_SOLVER_2D : public HMF_MEANPASS_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const float* rx;
    const float* ry;
    const float* rx_b;
    const float* ry_b;
    float* alloc;

protected:
    int min_iter_calc();
    void init_reg_info();
    void clean_up();
    void update_spatial_flow_calc();
    void parity_mask_buffer(float* buffer, const int parity);
    void parity_merge_buffer(float* buffer, const float* other, const int parity);
    
public:
    ~HMF_MEANPASS_CPU_SOLVER_2D();
    HMF_MEANPASS_CPU_SOLVER_2D(
        const bool channels_first,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_c,
        const int n_r,
        const int sizes[2],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
		const float* init_u,
        float* u );
};

class HMF_MEANPASS_CPU_GRADIENT_2D : public HMF_MEANPASS_CPU_GRADIENT_BASE
{
private:
    const int n_x;
    const int n_y;
    float* g_rx;
    float* g_ry;
	const float* rx;
	const float* ry;
    const float* rx_b;
    const float* ry_b;
    float* alloc;

protected:
    int min_iter_calc();
    void init_reg_info();
    void clean_up();
    void get_reg_gradients_and_push(float tau);
    
public:
    ~HMF_MEANPASS_CPU_GRADIENT_2D();
    HMF_MEANPASS_CPU_GRADIENT_2D(
        const bool channels_first,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_c,
        const int n_r,
        const int sizes[2],
        const float* u,
        const float* g,
		const float* rx_cost,
		const float* ry_cost,
        float* g_d,
        float* g_rx,
        float* g_ry );
};

#endif
