    

#ifndef HMF_MEANPASS_GPU_SOLVER_H
#define HMF_MEANPASS_GPU_SOLVER_H

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "hmf_trees.h"

using GPUDevice = Eigen::GpuDevice;

class HMF_MEANPASS_GPU_SOLVER_BASE
{
private:

protected:
    const GPUDevice & dev;
    TreeNode const* const* bottom_up_list;
    const int b;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* const data;
    float* const u;
    float* const temp;
    
    // optimization constants
    const float tau = 0.5f;
    const float beta = 0.005f;
    const float epsilon = 10e-5f;
    
    virtual int min_iter_calc() = 0;
    virtual void update_spatial_flow_calc() = 0;
    void block_iter();
    
public:
    HMF_MEANPASS_GPU_SOLVER_BASE(
        const GPUDevice & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_s,
        const int n_c,
        const int n_r,
        const float* const data_cost,
        float* const u,
        float** full_buff,
        float** img_buff);
    
    void operator()();
};

class HMF_MEANPASS_GPU_GRADIENT_BASE
{
private:

protected:
    const GPUDevice & dev;
    TreeNode const* const* bottom_up_list;
    const int b;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* const u;
    const float* const grad;
    float* const g_data;
    float* const u_tmp;
    float* const tmp;
    float* const dy;
    float* const du;
    
    // optimization constants
    const float tau = 0.5f;
    const float beta = 0.005f;
    const float epsilon = 10e-5f;
    
    virtual int min_iter_calc() = 0;
    virtual void update_spatial_flow_calc() = 0;
    
public:
    HMF_MEANPASS_GPU_GRADIENT_BASE(
        const GPUDevice & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int n_s,
        const int n_c,
        const int n_r,
        const float* const u,
        const float* const g,
        float* const g_d,
        float** full_buff);
    
    void operator()();
};

#endif //HMF_MEANPASS_GPU_SOLVER_H
