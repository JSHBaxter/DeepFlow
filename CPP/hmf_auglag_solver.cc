
#include "hmf_auglag_solver.h"
#include "spatial_star_auglag.h"
#include "hmf_trees.h"
#include <iostream>
#include <limits>

#include "common.h"
#include "cpu_kernels.h"
#ifdef USE_CUDA
#include "gpu_kernels.h"
#endif

template<typename DEV>
HMF_AUGLAG_SOLVER<DEV>::HMF_AUGLAG_SOLVER(
    const DEV & dev,
    TreeNode** bottom_up_list,
    const bool star,
    const int dim,
    const int* dims,
    const int n_c,
    const int n_r,
    const float * const * const inputs,
    float* u) :
MAXFLOW_ALGORITHM<DEV>(dev),
bottom_up_list(bottom_up_list),
n_c(n_c),
n_r(n_r),
n_s(product(dim,dims)),
data(inputs[0]),
u(u),
pt(0),
u_tmp(0),
div(0),
g(0),
ps(0),
g_ps(0),
ps_ind(0),
g_ind(0),
num_children(0)
{
    if(DEBUG_ITER) std::cout << "HMF_AUGLAG_SOLVER Constructor " << dim << " " << n_s << " " << n_c << " " << n_r << " " << std::endl;
    if(star)
        spatial_flow = new SPATIAL_STAR_AUGLAG_SOLVER<DEV>(dev, n_r, dims, dim, inputs+1);
    else
        spatial_flow = new SPATIAL_AUGLAG_SOLVER<DEV>(dev, n_r, dims, dim, inputs+1);
    MAXFLOW_ALGORITHM<DEV>::construct();
}

template<typename DEV>
void HMF_AUGLAG_SOLVER<DEV>::allocate_buffers(float* buffer, float** const carry_over, const float** const c_carry_over){
    if(DEBUG_ITER) std::cout << "HMF_AUGLAG_SOLVER Allocate " << std::endl;
    ps = buffer;
    g_ps = ps + n_s;
    pt = g_ps + n_s;
    div = pt + n_s*n_r;
    g = div + n_s*n_r;
    u_tmp = g + n_s*n_r;
    float* carry_over_tmp[] = {g, div};
    spatial_flow->allocate_buffers(u_tmp+n_r*n_s, carry_over_tmp, 0);

    //get pointer to parents' sink buffer (for fetching source flows)
    ps_ind = (float**) allocate_memory(MAXFLOW_ALGORITHM<DEV>::dev, 2*n_r*(sizeof(float*)+sizeof(float))); //get some custom memory (more robust to type size changes)
    float** tmp_ps_ind = new float* [n_r];
    for(int n_n = 0; n_n < n_r; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        if(n->parent->parent == NULL)
            tmp_ps_ind[n->r] = ps;
        else
            tmp_ps_ind[n->r] = pt + n_s*n->parent->r;
    }
    move_memory_to_device(MAXFLOW_ALGORITHM<DEV>::dev, tmp_ps_ind, ps_ind, n_r*sizeof(float*));
    delete [] tmp_ps_ind;
    
    //get pointer to parents' g buffer (for updating flows)
    g_ind = ps_ind + n_r;
    float** tmp_g_ind = new float* [n_r];
    for(int n_n = 0; n_n < n_r; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        if(n->parent->parent == NULL)
            tmp_g_ind[n->r] = g_ps;
        else
            tmp_g_ind[n->r] = g + n_s*n->parent->r;
    }
    move_memory_to_device(MAXFLOW_ALGORITHM<DEV>::dev, tmp_g_ind, g_ind, n_r*sizeof(float*));
    delete [] tmp_g_ind;
    
    //get number of children (save as float)
    num_children = (float*)((void*)(g_ind + n_r));
    float* tmp_c = new float [n_r];
    for(int n_n = 0; n_n < n_r; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        tmp_c[n->r] = (float) n->c;
    }
    move_memory_to_device(MAXFLOW_ALGORITHM<DEV>::dev, tmp_c, num_children, n_r*sizeof(float));
    delete [] tmp_c;
    
}

template<typename DEV>
int  HMF_AUGLAG_SOLVER<DEV>::get_buffer_size(){
    return 2*n_s + 4*n_s*n_r + spatial_flow->get_buffer_size();
}

template<typename DEV>
HMF_AUGLAG_SOLVER<DEV>::~HMF_AUGLAG_SOLVER(){
    delete spatial_flow;
    deallocate_memory(MAXFLOW_ALGORITHM<DEV>::dev, (void**) &ps_ind);
}

template<typename DEV>
void HMF_AUGLAG_SOLVER<DEV>::block_iter(){
    
    //calculate the capacity for the spatial flows
    //calc_capacity_hmf(dev, g, ps_ind, div, pt, u_tmp, n_s, n_r, icc, tau);
    if(DEBUG_PRINT){ std::cout << "ps: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,ps,n_s); }
    if(DEBUG_PRINT){ std::cout << "pt: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,pt,n_s*n_r); }
    if(DEBUG_PRINT){ std::cout << "u_tmp: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,u_tmp,n_s*n_r); }
    for(int n_n = 0; n_n < n_r; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        int r = n->r;
        if( n->parent->parent == NULL ){
            //std::cout << "Calc capacities" << g+r*n_s << " " << u_tmp+r*n_s << " " << ps << " " << pt+r*n_s << " " << div+r*n_s << " " << std::endl;
            calc_capacity_binary(MAXFLOW_ALGORITHM<DEV>::dev, g+r*n_s, div+r*n_s, ps, pt+r*n_s, u_tmp+r*n_s, n_s, icc, tau);
        }else{
            //std::cout << "Calc capacities" << g+r*n_s << " " << u_tmp+r*n_s << " " << pt+n->parent->r*n_s << " " << pt+r*n_s << " " << div+r*n_s << " " << std::endl;
            calc_capacity_binary(MAXFLOW_ALGORITHM<DEV>::dev, g+r*n_s, div+r*n_s, pt+n->parent->r*n_s, pt+r*n_s, u_tmp+r*n_s, n_s, icc, tau);
        }
    }
    if(DEBUG_PRINT){ std::cout << "g: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,g,n_s*n_r); }
    
    //update spatial flows
    spatial_flow->run();
    if(DEBUG_PRINT){ std::cout << "div: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,div,n_s*n_r); }
        
    //update source and sink multipliers bottom up
    for(int n_n = 0; n_n < n_r+1; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        float* n_pt_buf = pt+n->r*n_s;
        float* n_g_buf = g+n->r*n_s;
        float* n_div_buf = div+n->r*n_s;
        float* n_u_buf = u_tmp+n->r*n_s;
        
        //completed all leaves so constrain them
        if(n_n == n_c)
            max_neg_constrain(MAXFLOW_ALGORITHM<DEV>::dev, pt+n_s*(n_r-n_c), data, n_s*n_c);
    
        //if we are the source node
        if(n->r == -1){
            //std::cout << "Source " << n->r << " " << n->d << std::endl;
            set_buffer(MAXFLOW_ALGORITHM<DEV>::dev, ps, icc, n_s);
            //std::cout << "ps\t";  if(DEBUG_PRINT) print_buffer(dev, ps, n_s);
            for(int c = 0; c < n->c; c++){
                const TreeNode* nc = n->children[c];
                float* c_pt_buf = pt+nc->r*n_s;
                float* c_div_buf = div+nc->r*n_s;
                float* c_u_buf = u_tmp+nc->r*n_s;
                inc_inc_minc_buffer(MAXFLOW_ALGORITHM<DEV>::dev, c_pt_buf, c_div_buf, c_u_buf, -icc, ps, n_s);
                //std::cout << "c_pt\t";  if(DEBUG_PRINT) print_buffer(dev, c_pt_buf, n_s);
                //std::cout << "c_div\t"; if(DEBUG_PRINT) print_buffer(dev, c_div_buf, n_s);
                //std::cout << "c_u\t";   if(DEBUG_PRINT) print_buffer(dev, c_u_buf, n_s);
                //std::cout << "ps\t";  if(DEBUG_PRINT) print_buffer(dev, ps, n_s);
            }
            mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, 1.0f / (float) n->c, ps, n_s);
            //std::cout << "ps\t";  if(DEBUG_PRINT) print_buffer(dev, ps, n_s);
            //std::cout << std::endl;
        }

        //if we are a branch node
        else if(n->c > 0){
            //std::cout << "Branch " << n->r << " " << n->d << std::endl;
            const TreeNode* p = n->parent;
            float* p_pt_buf = pt+p->r*n_s;
            if( p->r == -1 )
                p_pt_buf = ps;
            copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev, p_pt_buf, n_pt_buf, n_s);
            ninc_buffer(MAXFLOW_ALGORITHM<DEV>::dev, n_div_buf, n_pt_buf, n_s);
            inc_mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, n_u_buf, n_pt_buf, n_s, icc);
            //std::cout << "p_pt\t";  if(DEBUG_PRINT) print_buffer(dev, p_pt_buf, n_s);
            //std::cout << "n_div\t"; if(DEBUG_PRINT) print_buffer(dev, n_div_buf, n_s);
            //std::cout << "n_u\t";   if(DEBUG_PRINT) print_buffer(dev, n_u_buf, n_s);
            //std::cout << "n_pt\t";  if(DEBUG_PRINT) print_buffer(dev, n_pt_buf, n_s);
            for(int c = 0; c < n->c; c++){
                const TreeNode* nc = n->children[c];
                float* c_pt_buf = pt+nc->r*n_s;
                float* c_div_buf = div+nc->r*n_s;
                float* c_u_buf = u_tmp+nc->r*n_s;
                //std::cout << "c_pt\t";  if(DEBUG_PRINT) print_buffer(dev, c_pt_buf, n_s);
                //std::cout << "c_div\t"; if(DEBUG_PRINT) print_buffer(dev, c_div_buf, n_s);
                //std::cout << "c_u\t";   if(DEBUG_PRINT) print_buffer(dev, c_u_buf, n_s);
                inc_inc_minc_buffer(MAXFLOW_ALGORITHM<DEV>::dev, c_pt_buf, c_div_buf, c_u_buf, -icc, n_pt_buf, n_s);
                //std::cout << "n_pt\t";  if(DEBUG_PRINT) print_buffer(dev, n_pt_buf, n_s);
            }
            mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, 1.0f / (float) (n->c+1), n_pt_buf, n_s);
            //std::cout << "n_pt\t";  if(DEBUG_PRINT) print_buffer(dev, n_pt_buf, n_s);
            //std::cout << std::endl;
        }
    
        //if we are a leaf node
        else{
            //std::cout << "Leaf " << n->r << " " << n->d << std::endl;
            const TreeNode* p = n->parent;
            float* p_pt_buf = pt+p->r*n_s;
            if( p->r == -1 )
                p_pt_buf = ps;
            copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev, p_pt_buf, n_pt_buf, n_s);
            ninc_buffer(MAXFLOW_ALGORITHM<DEV>::dev, n_div_buf, n_pt_buf, n_s);
            inc_mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, n_u_buf, n_pt_buf, n_s, icc);
            //std::cout << "p_pt\t";  if(DEBUG_PRINT) print_buffer(dev, p_pt_buf, n_s);
            //std::cout << "n_div\t"; if(DEBUG_PRINT) print_buffer(dev, n_div_buf, n_s);
            //std::cout << "n_u\t";   if(DEBUG_PRINT) print_buffer(dev, n_u_buf, n_s);
            //std::cout << "n_pt\t";  if(DEBUG_PRINT) print_buffer(dev, n_pt_buf, n_s);
            //std::cout << std::endl;
    
        }
    }
    if(DEBUG_PRINT){ std::cout << "ps: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,ps,n_s); }
    if(DEBUG_PRINT){ std::cout << "pt: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,pt,n_s*n_r); }
    
    //update multipliers
    //update_multiplier_hmf(dev, ps_ind, div, pt, u_tmp, g, n_s, n_r, cc);
    copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev, pt, g, n_s*n_r);
    inc_buffer(MAXFLOW_ALGORITHM<DEV>::dev, div, g, n_s*n_r);
    for(int n_n = 0; n_n < n_r; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        const TreeNode* p = n->parent;
        float* n_g_buf = g+n->r*n_s;
        float* p_pt_buf = pt+p->r*n_s;
        if( p->r == -1 )
            p_pt_buf = ps;
        ninc_buffer(MAXFLOW_ALGORITHM<DEV>::dev, p_pt_buf, n_g_buf, n_s);
    }
    mult_buffer(MAXFLOW_ALGORITHM<DEV>::dev, -cc, g, n_s*n_r);
    //std::cout << "Updates " << cc << std::endl;
    //std::cout << std::endl;
    
    if(DEBUG_PRINT){ std::cout << "g: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,g,n_s*n_r); }
    inc_buffer(MAXFLOW_ALGORITHM<DEV>::dev, g, u_tmp, n_s*n_r);
    if(DEBUG_PRINT){ std::cout << "u_tmp: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,u_tmp,n_s*n_r); }
                 
    //std::cout << "Printing flows" << std::endl;
    //for(int n_n = 0; n_n < n_r+1; n_n++){
    //    const TreeNode* n = bottom_up_list[n_r-n_n];
    //    float* n_pt_buf = pt+n->r*n_s;
    //    if(n->r == -1)
    //        if(DEBUG_PRINT) print_buffer(dev, ps, n_s);
    //    else
    //        if(DEBUG_PRINT) print_buffer(dev, n_pt_buf, n_s);
    //}
    //std::cout << std::endl;
}

template<typename DEV>
void HMF_AUGLAG_SOLVER<DEV>::run(){
    if(DEBUG_ITER) std::cout << "HMF_AUGLAG_SOLVER Init" << std::endl;
    int u_tmp_offset = n_s*(n_r-n_c);
    float prior_max_change = std::numeric_limits<float>::infinity();
    if(DEBUG_PRINT){ std::cout << "data: "; print_buffer(MAXFLOW_ALGORITHM<DEV>::dev,data,n_s*n_c); }
    
    //initialize flows and labels
    spatial_flow->init();
    clear_buffer(MAXFLOW_ALGORITHM<DEV>::dev, div, 2*n_s*n_r+n_s*(n_r-n_c)); //div,g, and the first part of u_tmp
    init_flows_potts(MAXFLOW_ALGORITHM<DEV>::dev, data, ps, pt, u_tmp+u_tmp_offset, n_s, n_c);
    for (int l = n_c; l < n_r; l++)
        copy_buffer(MAXFLOW_ALGORITHM<DEV>::dev,ps,pt+l*n_s,n_s);
    for (int l = n_c; l < n_r; l++) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u_tmp+n->children[c]->r*n_s, u_tmp+n->r*n_s, n_s);
    }
    
    // iterate in blocks
    int min_iter = spatial_flow->get_min_iter() + n_r - n_c;
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = spatial_flow->get_max_iter();
    if (max_loop < 10)
        max_loop = 10;
    
    for(int i = 0; i < max_loop; i++){    
        
        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

        //Determine if converged
        float max_change = spat_max_of_buffer(MAXFLOW_ALGORITHM<DEV>::dev, g, n_s, n_r);
        if(DEBUG_ITER) std::cout << "HMF_AUGLAG_SOLVER Iter " << i << ": " << max_change << "\t" << prior_max_change-max_change << std::endl;
        if (abs(prior_max_change-max_change) < tau*beta && max_change < tau*beta)
            break;
        
        prior_max_change = max_change;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();
    
    //get final output
    log_buffer(MAXFLOW_ALGORITHM<DEV>::dev, u_tmp+u_tmp_offset, u, n_s*n_c);

    //clean up
    spatial_flow->deinit();
}

