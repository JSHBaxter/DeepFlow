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

class HMF_MEANPASS_GPU_SOLVER_3D : public HMF_MEANPASS_GPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const int n_z;
    const float* rx;
    const float* ry;
    const float* rz;
    
protected:
    int min_iter_calc(){
        return n_x + n_y + n_z;
    }
    
    void update_spatial_flow_calc(){
        get_effective_reg(dev, temp, u_full, rx, ry, rz, n_x, n_y, n_z, n_r);
    }

public:
    HMF_MEANPASS_GPU_SOLVER_3D(
        const GPUDevice & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[7],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
        float* const u,
        float** full_buff,
        float** img_buff) :
    HMF_MEANPASS_GPU_SOLVER_BASE(dev,
                                 bottom_up_list,
                                 batch,
                                 sizes[2]*sizes[3]*sizes[4],
                                 sizes[1],
                                 sizes[5],
                                 data_cost,
                                 u,
                                 full_buff,
                                 img_buff),
    n_x(sizes[2]),
    n_y(sizes[3]),
    n_z(sizes[4]),
    rx(rx_cost),
    ry(ry_cost),
    rz(rz_cost)
    {}

};
class HMF_MEANPASS_GPU_GRADIENT_3D : public HMF_MEANPASS_GPU_GRADIENT_BASE
{
private:
    const int n_x;
    const int n_y;
    const int n_z;
    float* const g_rx;
    float* const g_ry;
    float* const g_rz;
    const float* const rx;
    const float* const ry;
    const float* const rz;
    
protected:
    int min_iter_calc(){
        return n_x + n_y + n_z;
    }
	
	void clear_variables(){
		clear_buffer(dev, g_rx, n_s*n_r);
		clear_buffer(dev, g_ry, n_s*n_r);
		clear_buffer(dev, g_rz, n_s*n_r);
	}
    
    void update_spatial_flow_calc(){
		populate_reg_mean_gradients_and_add(dev, dy, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_r);
		get_gradient_for_u(dev, dy, dy, rx, ry, rz, n_x, n_y, n_z, n_r);
    }

public:
    HMF_MEANPASS_GPU_GRADIENT_3D(
        const GPUDevice & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[7],
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
        const float* u,
        const float* g,
        float* g_d,
        float* g_rx,
        float* g_ry,
        float* g_rz,
        float** full_buff) :
    HMF_MEANPASS_GPU_GRADIENT_BASE(dev,
                                 bottom_up_list,
                                 batch,
                                 sizes[2]*sizes[3]*sizes[4],
                                 sizes[1],
                                 sizes[5],
                                 u,
                                 g,
                                 g_d,
                                 full_buff),
    n_x(sizes[2]),
    n_y(sizes[3]),
    n_z(sizes[4]),
    rx(rx_cost),
    ry(ry_cost),
    rz(rz_cost),
    g_rx(g_rx),
    g_ry(g_ry),
    g_rz(g_rz)
    {}

};

template <>
struct HmfMeanpass3dFunctor<GPUDevice> {
    void operator()(
        const GPUDevice& d,
        int sizes[7],
        const int* parentage_g,
        const int* data_index_g,
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
        float* u,
        float** full_buff,
        float** img_buff){

        int n_s = sizes[2]*sizes[3]*sizes[4];
        int n_c = sizes[1];
        int n_r = sizes[5];
		
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
        free(parentage);
        free(data_index);
        //node->print_tree();
        //TreeNode::print_list(bottom_up_list, sizes[5]+1);

        int n_batches = sizes[0];
        for(int b = 0; b < n_batches; b++)
            HMF_MEANPASS_GPU_SOLVER_3D(d, bottom_up_list, b, sizes,
                                       data_cost + b*n_s*n_c,
                                       rx_cost + b*n_s*n_r,
                                       ry_cost + b*n_s*n_r,
                                       rz_cost + b*n_s*n_r,
                                       u + b*n_s*n_c,
                                       full_buff,
                                       img_buff)();

        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);

    }

    int num_buffers_full(){ return 2; }
    int num_buffers_images(){ return 0; }
    int num_buffers_branch(){ return 0; }
    int num_buffers_data(){ return 0; }
};

template <>
struct HmfMeanpass3dGradFunctor<GPUDevice>{
    void operator()(
        const GPUDevice& d,
        int sizes[7],
        const int* parentage_g,
        const int* data_index_g,
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
        const float* u,
        const float* g,
        float* g_data,
        float* g_rx,
        float* g_ry,
        float* g_rz,
        int* g_par,
        int* g_didx,
        float** full_buff,
        float** img_buff){

        //clear unusable derviative
        clear_buffer(d, g_par, sizes[5]);
        clear_buffer(d, g_didx, sizes[5]);

        int n_s = sizes[2]*sizes[3]*sizes[4];
        int n_c = sizes[1];
        int n_r = sizes[5];
		
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
        free(parentage);
        free(data_index);
        //node->print_tree();
        //TreeNode::print_list(bottom_up_list, sizes[5]+1);

        int n_batches = sizes[0];
        for(int b = 0; b < n_batches; b++)
            HMF_MEANPASS_GPU_GRADIENT_3D(d, bottom_up_list, b, sizes,
                                         rx_cost + b*n_s*n_r,
                                         ry_cost + b*n_s*n_r,
                                         rz_cost + b*n_s*n_r,
                                         u + b*n_s*n_c,
                                         g + b*n_s*n_c,
                                         g_data + b*n_s*n_c,
                                         g_rx + b*n_s*n_r,
                                         g_ry + b*n_s*n_r,
                                         g_rz + b*n_s*n_r,
                                         full_buff)();

        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);


    }
    int num_buffers_full(){ return 4; }
    int num_buffers_images(){ return 0; }
    int num_buffers_branch(){ return 0; }
    int num_buffers_data(){ return 0; }
};

#endif //GOOGLE_CUDA
