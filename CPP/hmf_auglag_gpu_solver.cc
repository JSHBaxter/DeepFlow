#ifdef GOOGLE_CUDA

#include "hmf_auglag_gpu_solver.h"
#include "hmf_trees.h"
#include "gpu_kernels.h"

void HMF_AUGLAG_GPU_SOLVER_BASE::block_iter(){
    
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
    update_spatial_flow_calc();

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
                inc_buffer(dev, pt+nc->r*n_s, ps, n_s);
                inc_buffer(dev, div+nc->r*n_s, ps, n_s);
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

HMF_AUGLAG_GPU_SOLVER_BASE::HMF_AUGLAG_GPU_SOLVER_BASE(
    const GPUDevice & dev,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_s,
    const int n_c,
    const int n_r,
    const float* data_cost,
    float* u,
    float** full_buff,
    float** img_buff) :
dev(dev),
bottom_up_list(bottom_up_list),
b(batch),
n_c(n_c),
n_r(n_r),
n_s(n_s),
data_b(data_cost+batch*n_s*n_c),
u_b(u+batch*n_s*n_c),
pt(full_buff[0]),
u_tmp(full_buff[1]),
div(full_buff[2]),
g(full_buff[3]),
ps(img_buff[0])
{}

void HMF_AUGLAG_GPU_SOLVER_BASE::operator()(){

    // optimization constants
    const float beta = 0.02f;
    const float epsilon = 10e-5f;

    //initialize variables
    clear_buffer(dev, u_tmp, n_s*n_r);
    clear_spatial_flows();
    clear_buffer(dev, div, n_s*n_r);
    clear_buffer(dev, pt, n_s*n_r);
    find_min_constraint(dev, ps, data_b, n_c, n_s);

    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10 )
        min_iter = 10;
    int max_loop = 200;

    for(int i = 0; i < max_loop; i++){    
        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++){
            std::cout << "Iter " << i << std::endl;
            //block_iter();
        }

        //Determine if converged
        //std::cout << "Thread #:" << b << "\tIter #: " << iter << " \tMax change: " << max_change << std::endl;
        float max_change = max_of_buffer(dev, g, n_s*n_c);
        std::cout << "Calculate max change: " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    //for (int iter = 0; iter < min_iter; iter++)
    //    block_iter();

    //get final output
    log_buffer(dev, u_tmp, u_b, n_s*n_c);

}

#endif