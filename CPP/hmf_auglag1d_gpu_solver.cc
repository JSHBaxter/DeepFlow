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

namespace HMF1DAL_GPU {
    
class SolverBatchThreadChannelsFirst
{
private:
    const GPUDevice & dev;
    TreeNode const* const* bottom_up_list;
    const int b;
    const int n_x;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* data_b;
    const float* rx_b;
    float* u_b;
    float* ps;
    float* pt;
    float* px;
    float* u_tmp;
    float* div;
    float* g;
    
    // optimization constants
    const float tau = 0.1f;
    const float beta = 0.05f;
    const float epsilon = 10e-5f;
    const float cc = 0.1;
    const float icc = 1.0f/cc;
    
public:
    SolverBatchThreadChannelsFirst(
        const GPUDevice & dev,
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[5],
        const float* data_cost,
        const float* rx_cost,
        float* u,
        float** full_buff,
        float** img_buff) :
    dev(dev),
    bottom_up_list(bottom_up_list),
    b(batch),
    n_x(sizes[2]),
    n_c(sizes[1]),
    n_r(sizes[3]),
    n_s(sizes[2]),
    data_b(data_cost+batch*sizes[2]*sizes[1]),
    rx_b(rx_cost+batch*sizes[2]*sizes[3]),
    u_b(u+batch*sizes[2]*sizes[1]),
    pt(full_buff[0]),
    px(full_buff[1]),
    u_tmp(full_buff[2]),
    div(full_buff[3]),
    g(full_buff[4]),
    ps(img_buff[0])
    {}
    
    void block_iter(){
        //calculate the capacity and then update flows
        std::cout << "\tUpdate capacity" << std::endl;
        for(int n_n = 0; n_n < n_r; n_n++){
            const TreeNode* n = bottom_up_list[n_n];
            int r = n->r;
            if( n->parent->parent == NULL )
                calc_capacity_potts(dev, g+r*n_s, div+r*n_s, ps, pt+r*n_s, u_tmp+r*n_s, n_s, 1, icc, tau);
            else
                calc_capacity_potts(dev, g+r*n_s, div+r*n_s, pt+n->parent->r*n_s, pt+r*n_s, u_tmp+r*n_s, n_s, 1, icc, tau);
        }
        std::cout << "\tUpdate flow" << std::endl;
        update_spatial_flows(dev, g, div, px, rx_b, n_x, n_r*n_s);

        std::cout << "\tUpdate source/sink flows" << std::endl;
        //update source and sink multipliers top down
        for(int n_n = 0; n_n < n_r+1; n_n++){
            const TreeNode* n = bottom_up_list[n_r-n_n];

            //if we are the source node
            if(n->r == -1){
                set_buffer(dev, ps, icc, n_s);
                for(int c = 0; c < n->c; c++){
                    const TreeNode* nc = n->children[c];
                    inc_buffer(dev, pt+nc->r*n_s, ps, n_s);
                    inc_buffer(dev, div+nc->r*n_s, ps, n_s);
                    inc_mult_buffer(dev, u_tmp+nc->r*n_s, ps, n_s, -icc);
                }
                div_buffer(dev, (float) n->c, ps, n_s);
            }
            
            //if we are a branch node
            else if(n->c > 0){
                const TreeNode* p = n->parent;
                if( p->r == -1 )
                    copy_buffer(dev, ps,pt+n->r*n_s,n_s);
                else
                    copy_buffer(dev, pt+p->r*n_s,pt+n->r*n_s,n_s);
                inc_mult_buffer(dev, u_tmp+n->r*n_s, ps, n_s, icc);
                ninc_buffer(dev, div+n->r*n_s, ps, n_s);
                for(int c = 0; c < n->c; c++){
                    const TreeNode* nc = n->children[c];
                    inc(pt+nc->r*n_s, ps, n_s);
                    inc(div+nc->r*n_s, ps, n_s);
                    inc_mult_buffer(dev, u_tmp+nc->r*n_s, ps, n_s, -cc);
                }
                div_buffer(dev, (float) (n->c+1), pt+n->r*n_s, n_s);
            }

            //if we are a leaf node
            else{
                const TreeNode* p = n->parent;
                if( p->r == -1 )
                    copy_buffer(dev, ps,pt+n->r*n_s,n_s);
                else
                    copy_buffer(dev, pt+p->r*n_s,pt+n->r*n_s,n_s);
                inc_mult_buffer(dev, u_tmp+n->r*n_s, ps, n_s, icc);
                ninc_buffer(dev, div+n->r*n_s, ps, n_s);
                max_neg_constrain(dev, pt+n->r*n_s,data_b+n->r*n_s,n_s);
            }
        }

        //update multipliers
        std::cout << "\tUpdate multipliers" << std::endl;
        for(int n_n = 0; n_n < n_r; n_n++){
            const TreeNode* n = bottom_up_list[n_n];
            copy_buffer(dev, pt+n->r*n_s,g,n_s);
            inc_buffer(dev, div+n->r*n_s,g,n_s);
            if(n->parent->r == -1)
                ninc_buffer(dev,ps,g,n_s);
            else
                ninc_buffer(dev,pt+n->parent->r*n_s,g,n_s);
            mult_buffer(dev, -cc, g, n_s);
            inc_buffer(dev, g, u_tmp+n->r*n_s,n_s);
        }
    }
    
    void operator()(){
        
        // optimization constants
        const float beta = 0.02f;
        const float epsilon = 10e-5f;

        //initialize variables
        clear_buffer(dev, u_tmp, n_s*n_r);
        clear_buffer(dev, px, n_s*n_r);
        clear_buffer(dev, div, n_s*n_r);
        clear_buffer(dev, pt, n_s*n_r);
        find_min_constraint(dev, ps, data_b, n_c, n_s);
        

        // iterate in blocks
        int min_iter = 10;
        if (n_x > min_iter)
            min_iter = n_x;
        int max_loop = 200;
        
        for(int i = 0; i < max_loop; i++){    
            //run the solver a set block of iterations
            for (int iter = 0; iter < min_iter; iter++){
                std::cout << "Iter " << i << std::endl;
                block_iter();
            }

            //Determine if converged
            //std::cout << "Thread #:" << b << "\tIter #: " << iter << " \tMax change: " << max_change << std::endl;
            float max_change = max_of_buffer(dev, g, n_s*n_c);
            std::cout << "Calculate max change: " << max_change << std::endl;
            if (max_change < tau*beta)
                break;
        }
        
        //run one last block, just to be safe
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();
        
        //get final output
        log_buffer(dev, u_tmp, u_b, n_s*n_c);
        
    
    }
};


}

template <>
struct HmfAuglag1dFunctor<GPUDevice> {
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

        //build the tree
        TreeNode* node = NULL;
        TreeNode** children = NULL;
        TreeNode** bottom_up_list = NULL;
        TreeNode** top_down_list = NULL;
        int* parentage = new int[sizes[3]];
        int* data_index = new int[sizes[3]];
        get_from_gpu(d, parentage_g, parentage, sizes[4]*sizeof(int));
        get_from_gpu(d, data_index_g, data_index, sizes[4]*sizeof(int));
        TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, data_index, sizes[3], sizes[1]);
        free(parentage);
        free(data_index);
        //node->print_tree();
        //TreeNode::print_list(bottom_up_list, sizes[5]+1);

        int n_batches = sizes[0];
        for(int b = 0; b < n_batches; b++)
            HMF1DAL_GPU::SolverBatchThreadChannelsFirst(d, bottom_up_list, b, sizes, data_cost, rx_cost, u, full_buff, img_buff)();

        TreeNode::free_tree(node, children, bottom_up_list, top_down_list);

    }

    int num_buffers_full(){ return 5; }
    int num_buffers_images(){ return 1; }
    int num_buffers_branch(){ return 0; }
    int num_buffers_data(){ return 0; }
};

#endif //GOOGLE_CUDA
