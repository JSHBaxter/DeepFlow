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

class HMF_MEANPASS_GPU_SOLVER_1D : public HMF_MEANPASS_GPU_SOLVER_BASE
{
private:
    const int n_x;
    const float* rx;
    
protected:
    int min_iter_calc(){
        return n_x;
    }
    
    void update_spatial_flow_calc(){
        get_effective_reg(dev, temp, temp, rx, n_x, n_r);
    }

public:
    HMF_MEANPASS_GPU_SOLVER_1D(
        const GPUDevice & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[5],
        const float* data_cost,
        const float* rx_cost,
        float* const u,
        float** full_buff,
        float** img_buff) :
    HMF_MEANPASS_GPU_SOLVER_BASE(dev,
                                 bottom_up_list,
                                 batch,
                                 sizes[2],
                                 sizes[1],
                                 sizes[3],
                                 data_cost,
                                 u,
                                 full_buff,
                                 img_buff),
    n_x(sizes[2]),
    rx(rx_cost)
    {}

};
class HMF_MEANPASS_GPU_GRADIENT_1D : public HMF_MEANPASS_GPU_GRADIENT_BASE
{
private:
    const int n_x;
    float* const g_rx;
    const float* const rx;
    
protected:
    int min_iter_calc(){
        return n_x;
    }
    
    void update_spatial_flow_calc(){
        //and calculate gradients for the rest
        populate_reg_mean_gradients(dev, dy, u_tmp, g_rx, n_x, n_r);
        
        //get gradients for u terms (without diminish from last iteration - save to du_i)
        get_gradient_for_u(dev, dy, du, rx, n_x, n_r);
    }

public:
    HMF_MEANPASS_GPU_GRADIENT_1D(
        const GPUDevice & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[5],
        const float* rx_cost,
        const float* u,
        const float* g,
        float* g_d,
        float* g_rx,
        float** full_buff) :
    HMF_MEANPASS_GPU_GRADIENT_BASE(dev,
                                 bottom_up_list,
                                 batch,
                                 sizes[2],
                                 sizes[1],
                                 sizes[3],
                                 u,
                                 g,
                                 g_d,
                                 full_buff),
    n_x(sizes[2]),
    rx(rx_cost),
    g_rx(g_rx)
    {}

};

template <>
struct HmfMeanpass1dFunctor<GPUDevice> {
    void operator()(
        const GPUDevice& d,
        int sizes[5],
        const int* parentage_g,
        const int* data_index_g,
        const float* data_cost,
        const float* rx_cost,
        float* u,
        float** full_buff,
        float** img_buff){
			
        int n_s = sizes[2];
        int n_c = sizes[1];
        int n_r = sizes[3];

        //build the tree
        TreeNode* node = NULL;
        TreeNode** children = NULL;
        TreeNode** bottom_up_list = NULL;
        TreeNode** top_down_list = NULL;
        int* parentage = new int[n_r];
        int* data_index = new int[n_r];
        get_from_gpu(d, parentage_g, parentage, n_r*sizeof(int));
        get_from_gpu(d, data_index_g, data_index, n_r*sizeof(int));
        TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, data_index, n_r, n_c);
        delete parentage;
        delete data_index;
        //node->print_tree();
        //TreeNode::print_list(bottom_up_list, sizes[5]+1);

        int n_batches = sizes[0];
        for(int b = 0; b < n_batches; b++)
            HMF_MEANPASS_GPU_SOLVER_1D(d, bottom_up_list, b, sizes,
                                       data_cost + b*n_s*n_c,
                                       rx_cost + b*n_s*n_r,
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
struct HmfMeanpass1dGradFunctor<GPUDevice>{
    void operator()(
        const GPUDevice& d,
        int sizes[5],
        const int* parentage_g,
        const int* data_index_g,
        const float* data_cost,
        const float* rx_cost,
        const float* u,
        const float* g,
        float* g_data,
        float* g_rx,
        int* g_par,
        int* g_didx,
        float** full_buff,
        float** img_buff){

        //clear unusable derviative
        clear_buffer(d, g_par, sizes[3]);
        clear_buffer(d, g_didx, sizes[3]);

        //build the tree
        TreeNode* node = NULL;
        TreeNode** children = NULL;
        TreeNode** bottom_up_list = NULL;
        TreeNode** top_down_list = NULL;
        int* parentage = new int[sizes[3]];
        int* data_index = new int[sizes[3]];
        get_from_gpu(d, parentage_g, parentage, sizes[3]*sizeof(int));
        get_from_gpu(d, data_index_g, data_index, sizes[3]*sizeof(int));
        TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, data_index, sizes[5], sizes[1]);
        delete parentage;
        delete data_index;

        int n_batches = sizes[0];
        int n_s = sizes[2];
        int n_c = sizes[1];
        int n_r = sizes[3];
        for(int b = 0; b < n_batches; b++)
            HMF_MEANPASS_GPU_GRADIENT_1D(d, bottom_up_list, b, sizes,
                                         rx_cost + b*n_s*n_r,
                                         u + b*n_s*n_c,
                                         g + b*n_s*n_c,
                                         g_data + b*n_s*n_c,
                                         g_rx + b*n_s*n_r,
                                         full_buff)();

        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);


    }
    int num_buffers_full(){ return 4; }
    int num_buffers_images(){ return 0; }
    int num_buffers_branch(){ return 0; }
    int num_buffers_data(){ return 0; }
};

#endif //GOOGLE_CUDA
