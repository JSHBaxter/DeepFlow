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
#include "hmf_auglag_gpu_solver.h"
#include "gpu_kernels.h"
#include <algorithm>


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
    int min_iter_calc() {
        return std::max(n_x,n_y)+n_r-n_c;
    }
    
    virtual void clear_spatial_flows(){
        clear_buffer(dev, px, n_s*n_r);
        clear_buffer(dev, py, n_s*n_r);
    }
    
    virtual void update_spatial_flow_calc(){
        update_spatial_flows(dev, g, div, px, py, rx_b, ry_b, n_x, n_y, n_r*n_s);
    }
   
public:
    HMF_AUGLAG_GPU_SOLVER_2D(
        const GPUDevice & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[6],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        float* u,
        float** full_buff,
        float** img_buff) :
    HMF_AUGLAG_GPU_SOLVER_BASE(dev,
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
    rx_b(rx_cost),
    ry_b(rx_cost),
    px(full_buff[4]),
    py(full_buff[5])
    {}
};

template <>
struct HmfAuglag2dFunctor<GPUDevice> {
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

        int n_s = sizes[2]*sizes[3];
        int n_c = sizes[1];
        int n_r = sizes[4];
		
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
            HMF_AUGLAG_GPU_SOLVER_2D(d, bottom_up_list, b, sizes, 
                                     data_cost + b*n_s*n_c,
                                     rx_cost + b*n_s*n_r,
                                     ry_cost + b*n_s*n_r,
                                     u + b*n_s*n_c,
                                     full_buff, img_buff)();

        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);

    }

    int num_buffers_full(){ return 6; }
    int num_buffers_images(){ return 2; }
    int num_buffers_branch(){ return 0; }
    int num_buffers_data(){ return 0; }
};

#endif //GOOGLE_CUDA
