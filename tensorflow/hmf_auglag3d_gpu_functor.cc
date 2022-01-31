#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "../CPP/hmf_trees.h"
#include "../CPP/hmf_auglag3d_gpu_solver.h"

template <>
struct HmfAuglag3dFunctor<GPUDevice> {
    void operator()(
        const GPUDevice& d,
        int sizes[7],
        const int* parentage_g,
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
        int* parentage = new int[n_r];;
        get_from_gpu(d.stream(), parentage_g, parentage, n_r*sizeof(int));
        TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, n_r, n_c);
        delete parentage;
        //node->print_tree();
        //TreeNode::print_list(bottom_up_list, sizes[5]+1);

        int n_batches = sizes[0];
        int data_sizes[3] = {sizes[2],sizes[3],sizes[4]};
        for(int b = 0; b < n_batches; b++)
            HMF_AUGLAG_GPU_SOLVER_3D(d.stream(), bottom_up_list, b, n_c, n_r, data_sizes, 
                                     data_cost + b*n_s*n_c,
                                     rx_cost + b*n_s*n_r,
                                     ry_cost + b*n_s*n_r,
                                     rz_cost + b*n_s*n_r,
                                     u + b*n_s*n_c,
                                     full_buff, img_buff)();

        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);

    }

    int num_buffers_full(){ return HMF_AUGLAG_GPU_SOLVER_3D::num_buffers_full(); }
    int num_buffers_images(){ return HMF_AUGLAG_GPU_SOLVER_3D::num_buffers_images(); }
    int num_buffers_branch(){ return HMF_AUGLAG_GPU_SOLVER_3D::num_buffers_branch(); }
    int num_buffers_data(){ return HMF_AUGLAG_GPU_SOLVER_3D::num_buffers_data(); }
};

#endif //GOOGLE_CUDA
