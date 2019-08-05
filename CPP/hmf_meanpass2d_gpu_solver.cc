#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <math.h>
#include <thread>
#include <iostream>
#include <limits>
#include "hmf_trees.h"
#include "hmf_meanpass_gpu_solver.h"
#include "gpu_kernels.h"

class HMF_MEANPASS_GPU_SOLVER_2D : public HMF_MEANPASS_GPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const float* rx;
    const float* ry;
    
protected:
    int min_iter_calc(){
        return n_x + n_y;
    }
    
    void update_spatial_flow_calc(){
        get_effective_reg(dev, temp, temp, rx, ry, n_x, n_y, n_r);
    }

public:
    HMF_MEANPASS_GPU_SOLVER_2D(
        const GPUDevice & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[6],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        float* const u,
        float** full_buff,
        float** img_buff) :
    HMF_MEANPASS_GPU_SOLVER_BASE(dev,
                                 bottom_up_list,
                                 batch,
                                 sizes[2]*sizes[3],
                                 sizes[1],
                                 sizes[4],
                                 data_cost,
                                 u,
                                 full_buff,
                                 img_buff),
    n_x(sizes[2]),
    n_y(sizes[3]),
    rx(rx_cost),
    ry(ry_cost)
    {}

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
    int min_iter_calc(){
        return n_x + n_y;
    }
    
    void update_spatial_flow_calc(){
        //and calculate gradients for the rest
        populate_reg_mean_gradients(dev, dy, u_tmp, g_rx, g_ry, n_x, n_y, n_r);
        
        //get gradients for u terms (without diminish from last iteration - save to du_i)
        get_gradient_for_u(dev, dy, du, rx, ry, n_x, n_y, n_r);
    }

public:
    HMF_MEANPASS_GPU_GRADIENT_2D(
        const GPUDevice & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[6],
        const float* rx_cost,
        const float* ry_cost,
        const float* u,
        const float* g,
        float* g_d,
        float* g_rx,
        float* g_ry,
        float** full_buff) :
    HMF_MEANPASS_GPU_GRADIENT_BASE(dev,
                                 bottom_up_list,
                                 batch,
                                 sizes[2]*sizes[3],
                                 sizes[1],
                                 sizes[4],
                                 u,
                                 g,
                                 g_d,
                                 full_buff),
    n_x(sizes[2]),
    n_y(sizes[3]),
    rx(rx_cost),
    ry(ry_cost),
    g_rx(g_rx),
    g_ry(g_ry)
    {}

};

template <>
struct HmfMeanpass2dFunctor<GPUDevice> {
    void operator()(
        const GPUDevice& d,
        int sizes[6],
        const int* parentage_g,
        const int* data_index_g,
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        float* u,
        float** full_buff,
        float** img_buff){

        //build the tree
        TreeNode* node = NULL;
        TreeNode** children = NULL;
        TreeNode** bottom_up_list = NULL;
        TreeNode** top_down_list = NULL;
        int* parentage = new int[sizes[4]];
        int* data_index = new int[sizes[4]];
        get_from_gpu(d, parentage_g, parentage, sizes[4]*sizeof(int));
        get_from_gpu(d, data_index_g, data_index, sizes[4]*sizeof(int));
        TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, data_index, sizes[5], sizes[1]);
        delete parentage;
        delete data_index;
        //node->print_tree();
        //TreeNode::print_list(bottom_up_list, sizes[5]+1);

        int n_batches = sizes[0];
        int n_s = sizes[2]*sizes[3];
        int n_c = sizes[1];
        int n_r = sizes[4];
        for(int b = 0; b < n_batches; b++)
            HMF_MEANPASS_GPU_SOLVER_2D(d, bottom_up_list, b, sizes,
                                       data_cost + b*n_s*n_c,
                                       rx_cost + b*n_s*n_r,
                                       ry_cost + b*n_s*n_r,
                                       u + b*n_s*n_c,
                                       full_buff,
                                       img_buff)();

        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);

    }

    int num_buffers_full(){ return 1; }
    int num_buffers_images(){ return 0; }
    int num_buffers_branch(){ return 0; }
    int num_buffers_data(){ return 0; }
};

template <>
struct HmfMeanpass2dGradFunctor<GPUDevice>{
    void operator()(
        const GPUDevice& d,
        int sizes[6],
        const int* parentage_g,
        const int* data_index_g,
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        const float* u,
        const float* g,
        float* g_data,
        float* g_rx,
        float* g_ry,
        int* g_par,
        int* g_didx,
        float** full_buff,
        float** img_buff){

        //clear unusable derviative
        clear_buffer(d, g_par, sizes[4]);
        clear_buffer(d, g_didx, sizes[4]);

        //build the tree
        TreeNode* node = NULL;
        TreeNode** children = NULL;
        TreeNode** bottom_up_list = NULL;
        TreeNode** top_down_list = NULL;
        int* parentage = new int[sizes[4]];
        int* data_index = new int[sizes[4]];
        get_from_gpu(d, parentage_g, parentage, sizes[4]*sizeof(int));
        get_from_gpu(d, data_index_g, data_index, sizes[4]*sizeof(int));
        TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, data_index, sizes[5], sizes[1]);
        delete parentage;
        delete data_index;

        int n_batches = sizes[0];
        int n_s = sizes[2]*sizes[3];
        int n_c = sizes[1];
        int n_r = sizes[4];
        for(int b = 0; b < n_batches; b++)
            HMF_MEANPASS_GPU_GRADIENT_2D(d, bottom_up_list, b, sizes,
                                         rx_cost + b*n_s*n_r,
                                         ry_cost + b*n_s*n_r,
                                         u + b*n_s*n_c,
                                         g + b*n_s*n_c,
                                         g_data + b*n_s*n_c,
                                         g_rx + b*n_s*n_r,
                                         g_ry + b*n_s*n_r,
                                         full_buff)();

        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);


    }
    int num_buffers_full(){ return 4; }
    int num_buffers_images(){ return 0; }
    int num_buffers_branch(){ return 0; }
    int num_buffers_data(){ return 0; }
};

#endif //GOOGLE_CUDA