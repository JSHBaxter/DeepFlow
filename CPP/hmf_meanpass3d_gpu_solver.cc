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
#include "gpu_kernels.h"

namespace HMF_GPU {
    
class SolverBatchThreadChannelsFirst
{
private:
    const GPUDevice & dev;
    TreeNode const* const* bottom_up_list;
    const int b;
    const int n_x;
    const int n_y;
    const int n_z;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* data_b;
    const float* rx_b;
    const float* ry_b;
    const float* rz_b;
    float* u_b;
    float* temp;
    const float tau = 0.5f;
    
public:
    SolverBatchThreadChannelsFirst(
        const GPUDevice & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[7],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
        float* u,
        float** full_buff,
        float** img_buff) :
    dev(dev),
    bottom_up_list(bottom_up_list),
    b(batch),
    n_x(sizes[2]),
    n_y(sizes[3]),
    n_z(sizes[4]),
    n_c(sizes[1]),
    n_r(sizes[5]),
    n_s(sizes[2]*sizes[3]*sizes[4]),
    data_b(data_cost+batch*sizes[2]*sizes[3]*sizes[4]*sizes[1]),
    rx_b(rx_cost+batch*sizes[2]*sizes[3]*sizes[4]*sizes[5]),
    ry_b(ry_cost+batch*sizes[2]*sizes[3]*sizes[4]*sizes[5]),
    rz_b(rz_cost+batch*sizes[2]*sizes[3]*sizes[4]*sizes[5]),
    u_b(u+batch*sizes[2]*sizes[3]*sizes[4]*sizes[1]),
    temp(full_buff[0])
    {}
    
    void block_iter(){
        
        //calculate the aggregate probabilities (stored in temp)
        copy_buffer(dev, u_b, temp, n_s*n_c);
        clear_buffer(dev, temp+n_c*n_s, n_s*(n_r-n_c));
        for (int l = n_c; l < n_r; l++) {
            const TreeNode* n = bottom_up_list[l];
            for(int c = 0; c < n->c; c++)
                inc_buffer(dev, temp+n->children[c]->r*n_s, temp+n->r*n_s, n_s);
        }

        //calculate the effective regularization (overwrites own temp)
        get_effective_reg(dev, temp, temp, rx_b, ry_b, rz_b, n_x, n_y, n_z, n_r);

        //calculate the aggregate effective regularization (overwrites own temp)
        for (int l = n_r-1; l >= n_c; l--) {
            const TreeNode* n = bottom_up_list[l];
            for(int c = 0; c < n->c; c++)
                inc_buffer(dev, temp+n->r*n_s, temp+n->children[c]->r*n_s, n_s);
        }

        // get new probability estimates, and normalize (store answer in temp)
        softmax(dev, data_b, temp, temp, n_s, n_c);

        //update labels 
        change_to_diff(dev, u_b, temp, n_s*n_c, tau);
    }
    
    void operator()(){
        
        // optimization constants
        const float beta = 0.02f;
        const float epsilon = 10e-5f;

        //initialize variables
        softmax(dev, data_b, NULL, u_b, n_s, n_c);

        // iterate in blocks
        int min_iter = 10;
        if (n_x > min_iter)
            min_iter = n_x;
        if (n_y > min_iter)
            min_iter = n_y;
        if (n_z > min_iter)
            min_iter = n_z;
        int max_loop = 200;
        
        for(int i = 0; i < max_loop; i++){    
            //run the solver a set block of iterations
            for (int iter = 0; iter < min_iter; iter++)
                block_iter();

            //Determine if converged
            //std::cout << "Thread #:" << b << "\tIter #: " << iter << " \tMax change: " << max_change << std::endl;
            float max_change = max_of_buffer(dev, temp, n_s*n_c);
            if (max_change < tau*beta)
                break;
        }
        
        //run one last block, just to be safe
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

        //calculate the aggregate probabilities
        copy_buffer(dev, u_b, temp, n_s*n_c);
        clear_buffer(dev, temp+n_c*n_s, n_s*(n_r-n_c));
        for (int l = n_c; l < n_r; l++) {
            const TreeNode* n = bottom_up_list[l];
            for(int c = 0; c < n->c; c++)
                inc_buffer(dev, temp+n->children[c]->r*n_s, temp+n->r*n_s, n_s);
        }

        //calculate the effective regularization
        get_effective_reg(dev, temp, temp, rx_b, ry_b, rz_b, n_x, n_y, n_z, n_r);

        //calculate the aggregate effective regularization
        for (int l = n_r-1; l >= n_c; l--) {
            const TreeNode* n = bottom_up_list[l];
            for(int c = 0; c < n->c; c++)
                inc_buffer(dev, temp+n->r*n_s, temp+n->children[c]->r*n_s, n_s);
        }

        
        //get final output
        add_then_store(dev, data_b, temp, u_b, n_c*n_s);
        
    
    }
};


class GradientBatchThreadChannelsFirst
{
private:
    const GPUDevice & dev;
    TreeNode const* const* bottom_up_list;
    const int batch;
    const int n_x;
    const int n_y;
    const int n_z;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* rx_cost;
    const float* ry_cost;
    const float* rz_cost;
    float* g_data;
    float* g_rx;
    float* g_ry;
    float* g_rz;
    const float* u;
    const float* grad;
    float* du_i;
    float* u_tmp;
    float* dy;
    float* tmp;
    
public:
    GradientBatchThreadChannelsFirst(
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
    dev(dev),
    bottom_up_list(bottom_up_list),
    batch(batch),
    n_x(sizes[2]),
    n_y(sizes[3]),
    n_z(sizes[4]),
    n_c(sizes[1]),
    n_r(sizes[5]),
    n_s(sizes[2]*sizes[3]*sizes[4]),
    g_data(g_d),
    g_rx(g_rx),
    g_ry(g_ry),
    g_rz(g_rz),
    rx_cost(rx_cost),
    ry_cost(ry_cost),
    rz_cost(rz_cost),
    u(u),
    grad(g),
    u_tmp(full_buff[0]),
    dy(full_buff[1]),
    du_i(full_buff[2]),
    tmp(full_buff[3])
    {}
    
    void operator()(){
        
        // create easier pointers
        int b = this->batch;
        float* g_d_b = g_data + b*n_s*n_c;
        float* g_rx_b = g_rx + b*n_s*n_r;
        float* g_ry_b = g_ry + b*n_s*n_r;
        float* g_rz_b = g_rz + b*n_s*n_r;
        const float* rx_b = rx_cost + b*n_s*n_r;
        const float* ry_b = ry_cost + b*n_s*n_r;
        const float* rz_b = rz_cost + b*n_s*n_r;
        const float* g_b = grad + b*n_s*n_c;
        const float* u_b = u + b*n_s*n_c;

        const float epsilon = 10e-5f;
        const float beta = 1e-20;
        const float tau = 0.5;
        
        int min_iter = 10;
        if (n_x > min_iter)
            min_iter = n_x;
        if (n_y > min_iter)
            min_iter = n_y;
        if (n_z > min_iter)
            min_iter = n_z;
        int max_loop = 200;
        
        //calculate the aggregate probabilities
        softmax(dev, u_b, NULL, u_tmp, n_s, n_c);
        clear_buffer(dev, u_tmp+n_c*n_s, n_s*(n_r-n_c));
        for (int l = n_c; l < n_r; l++) {
            const TreeNode* n = bottom_up_list[l];
            for(int c = 0; c < n->c; c++)
                inc_buffer(dev, u_tmp+n->children[c]->r*n_s, u_tmp+n->r*n_s, n_s);
        }
        
        //calculate aggregate gradient
        copy_buffer_clip(dev, g_b, dy, n_s*n_c, 1.0f/(n_s*n_c));
        clear_buffer(dev, dy+n_s*n_c, n_s*(n_r-n_c));
        for (int l = n_c; l < n_r; l++) {
            const TreeNode* n = bottom_up_list[l];
            for(int c = 0; c < n->c; c++)
                inc_buffer(dev, dy+n->children[c]->r*n_s, dy+n->r*n_s, n_s);
        }
      
        // populate data gradient
        copy_buffer(dev, dy, g_d_b, n_s*n_c);
        
        //and calculate gradients for the rest
        populate_reg_mean_gradients(dev, dy, u_tmp, g_rx_b, g_ry_b, g_rz_b, n_x, n_y, n_z, n_r);
        
        //get gradients for u terms (without diminish from last iteration - save to du_i)
        get_gradient_for_u(dev, dy, du_i, rx_b, ry_b, rz_b, n_x, n_y, n_z, n_r);

        //collapse back down to leaves
        for (int l = n_r-1; l >= n_c; l--) {
            const TreeNode* n = bottom_up_list[l];
            for(int c = 0; c < n->c; c++)
                inc_buffer(dev, du_i+n->r*n_s, du_i+n->children[c]->r*n_s, n_s);
        }
        
        for(int i = 0; i < max_loop; i++){
            //push gradients back a number of iterations (first iteration has tau=1, the rest a smaller tau)
            for(int iter = 0; iter < min_iter; iter++){

                //process gradients and expand them upwards
                process_grad_potts(dev, du_i, u_tmp, dy, n_s, n_c, tau);
                clear_buffer(dev, dy+n_s*n_c, n_s*(n_r-n_c));
                for (int l = n_c; l < n_r; l++) {
                    const TreeNode* n = bottom_up_list[l];
                    for(int c = 0; c < n->c; c++)
                        inc_buffer(dev, dy+n->children[c]->r*n_s, dy+n->r*n_s, n_s);
                }

                //add into data term gradient
                inc_buffer(dev, dy, g_d_b, n_s*n_c);

                //get gradients for the regularization terms
                populate_reg_mean_gradients_and_add(dev, dy, u_tmp, g_rx_b, g_ry_b, g_rz_b, n_x, n_y, n_z, n_r);

                //get gradients for u terms (without diminish from last iteration - save to dy)
                get_gradient_for_u(dev, dy, tmp, rx_b, ry_b, rz_b, n_x, n_y, n_z, n_r);

                //collapse back down to leaves
                for (int l = n_r-1; l >= n_c; l--) {
                    const TreeNode* n = bottom_up_list[l];
                    for(int c = 0; c < n->c; c++)
                        inc_buffer(dev, tmp+n->r*n_s, tmp+n->children[c]->r*n_s, n_s);
                }

                //add diminished content from before
                mult_buffer(dev, 1.0f-tau-epsilon, du_i, n_s*n_c);
                inc_buffer(dev, tmp, du_i, n_c*n_s);
            }
            
            copy_buffer(dev, du_i, dy, n_s*n_c);
            float max_change = max_of_buffer(dev, dy, n_c*n_s);
            if(max_change < beta)
                break;
        }
        
        
    }
};


}


template struct HmfMeanpass3dFunctor<GPUDevice>;
template struct HmfMeanpass3dGradFunctor<GPUDevice>;

void HmfMeanpass3dFunctor<GPUDevice>::operator()(
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
    
    //build the tree
    TreeNode* node = NULL;
    TreeNode** children = NULL;
    TreeNode** bottom_up_list = NULL;
    TreeNode** top_down_list = NULL;
    int* parentage = new int[sizes[5]];
    int* data_index = new int[sizes[5]];
    get_from_gpu(d, parentage_g, parentage, sizes[5]*sizeof(int));
    get_from_gpu(d, data_index_g, data_index, sizes[5]*sizeof(int));
    TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, data_index, sizes[5], sizes[1]);
    free(parentage);
    free(data_index);
    //node->print_tree();
    //TreeNode::print_list(bottom_up_list, sizes[5]+1);

    int n_batches = sizes[0];
    for(int b = 0; b < n_batches; b++)
        HMF_GPU::SolverBatchThreadChannelsFirst(d, bottom_up_list, b, sizes, data_cost, rx_cost, ry_cost, rz_cost, u, full_buff, img_buff)();
      
    TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
      
}
int HmfMeanpass3dFunctor<GPUDevice>::num_buffers_full(){ return 1; }
int HmfMeanpass3dFunctor<GPUDevice>::num_buffers_images(){ return 0; }
int HmfMeanpass3dFunctor<GPUDevice>::num_buffers_branch(){ return 0; }
int HmfMeanpass3dFunctor<GPUDevice>::num_buffers_data(){ return 0; }

void HmfMeanpass3dGradFunctor<GPUDevice>::operator()(
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
    
    //build the tree
    TreeNode* node = NULL;
    TreeNode** children = NULL;
    TreeNode** bottom_up_list = NULL;
    TreeNode** top_down_list = NULL;
    int* parentage = new int[sizes[5]];
    int* data_index = new int[sizes[5]];
    get_from_gpu(d, parentage_g, parentage, sizes[5]*sizeof(int));
    get_from_gpu(d, data_index_g, data_index, sizes[5]*sizeof(int));
    TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, data_index, sizes[5], sizes[1]);
    free(parentage);
    free(data_index);
    
    int n_batches = sizes[0];
    for(int b = 0; b < n_batches; b++)
        HMF_GPU::GradientBatchThreadChannelsFirst(d, bottom_up_list, b, sizes, rx_cost, ry_cost, rz_cost, u, g, g_data, g_rx, g_ry, g_rz, full_buff)();
      
    TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
    

}
int HmfMeanpass3dGradFunctor<GPUDevice>::num_buffers_full(){ return 4; }
int HmfMeanpass3dGradFunctor<GPUDevice>::num_buffers_images(){ return 0; }
int HmfMeanpass3dGradFunctor<GPUDevice>::num_buffers_branch(){ return 0; }
int HmfMeanpass3dGradFunctor<GPUDevice>::num_buffers_data(){ return 0; }

#endif //GOOGLE_CUDA
