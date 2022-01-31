#ifndef HMF_AUGLAG2D_GPU_SOLVER_H
#define HMF_AUGLAG2D_GPU_SOLVER_H

#include "hmf_trees.h"
#include "hmf_auglag_gpu_solver.h"

class HMF_AUGLAG_GPU_SOLVER_2D : public HMF_AUGLAG_GPU_SOLVER_BASE
{
private:
    const float* rx_b;
    const float* ry_b;
    float* px;
    float* py;
    const int n_x;
    const int n_y;

protected:
    int min_iter_calc();
    virtual void clear_spatial_flows();
    virtual void update_spatial_flow_calc();
   
public:
    HMF_AUGLAG_GPU_SOLVER_2D(
        const cudaStream_t & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_c,
        const int n_r,
        const int sizes[2],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        float* u,
        float** full_buff,
        float** img_buff);
    
    static inline int num_buffers_full(){ return 6; }
    static inline int num_buffers_images(){ return 2; }
    static inline int num_buffers_branch(){ return 0; }
    static inline int num_buffers_data(){ return 0; }
};

#endif
