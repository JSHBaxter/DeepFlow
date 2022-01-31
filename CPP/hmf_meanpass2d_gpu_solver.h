#ifndef HMF_MEANPASS2D_GPU_SOLVER_H
#define HMF_MEANPASS2D_GPU_SOLVER_H

#include "hmf_trees.h"
#include "hmf_meanpass_gpu_solver.h"


class HMF_MEANPASS_GPU_SOLVER_2D : public HMF_MEANPASS_GPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const float* rx;
    const float* ry;
    
protected:
    int min_iter_calc();
    void update_spatial_flow_calc();
    void parity_mask_buffer(float* buffer, const int parity);
    void parity_merge_buffer(float* buffer, const float* other, const int parity);

public:
    HMF_MEANPASS_GPU_SOLVER_2D(
        const cudaStream_t & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_c,
        const int n_r,
        const int sizes[2],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
		const float* init_u,
        float* const u,
        float** full_buff,
        float** img_buff);

    static inline int num_buffers_full(){ return 2; }
    static inline int num_buffers_images(){ return 0; }
    static inline int num_buffers_branch(){ return 0; }
    static inline int num_buffers_data(){ return 0; }

};

class HMF_MEANPASS_GPU_GRADIENT_2D : public HMF_MEANPASS_GPU_GRADIENT_BASE
{
private:
    const int n_x;
    const int n_y;
    float* const g_rx;
    float* const g_ry;
    const float* const rx;
    const float* const ry;
    
protected:
    int min_iter_calc();
	void clear_variables();
    void get_reg_gradients_and_push(float tau);

public:
    HMF_MEANPASS_GPU_GRADIENT_2D(
        const cudaStream_t & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_c,
        const int n_r,
        const int sizes[2],
        const float* rx_cost,
        const float* ry_cost,
        const float* u,
        const float* g,
        float* g_d,
        float* g_rx,
        float* g_ry,
        float** full_buff);
        
    static inline int num_buffers_full(){ return 4; }
    static inline int num_buffers_images(){ return 0; }
    static inline int num_buffers_branch(){ return 0; }
    static inline int num_buffers_data(){ return 0; }

};


#endif
