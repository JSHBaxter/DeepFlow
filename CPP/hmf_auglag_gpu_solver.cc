#ifdef GOOGLE_CUDA

#include "hmf_auglag_gpu_solver.h"
#include "hmf_trees.h"
#include "gpu_kernels.h"

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
data(data_cost),
u(u),
pt(full_buff[0]),
u_tmp(full_buff[1]),
div(full_buff[2]),
g(full_buff[3]),
ps(img_buff[0]),
g_ps(img_buff[0])
{
    //std::cout << n_s << " " << n_c << " " << n_r << std::endl;
    //get pointer to parents' sink buffer (for fetching source flows)
    float** tmp_ps_ind = new float* [n_r];
    for(int n_n = 0; n_n < n_r; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        if(n->parent->parent == NULL)
            tmp_ps_ind[n->r] = ps;
        else
            tmp_ps_ind[n->r] = pt + n_s*n->parent->r;
    }
    
    ps_ind = (float**) allocate_on_gpu(dev, 2*n_r*sizeof(float*));
    send_to_gpu(dev, tmp_ps_ind, ps_ind, n_r*sizeof(float*));
    delete tmp_ps_ind;
    
    //get pointer to parents' g buffer (for updating flows)
    float** tmp_g_ind = new float* [n_r];
    for(int n_n = 0; n_n < n_r; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        if(n->parent->parent == NULL)
            tmp_g_ind[n->r] = g_ps;
        else
            tmp_g_ind[n->r] = g + n_s*n->parent->r;
    }
    
    g_ind = ps_ind + n_r;
    send_to_gpu(dev, tmp_g_ind, g_ind, n_r*sizeof(float*));
    delete tmp_g_ind;
    
    //get number of children
    int* tmp_c = new int [n_r];
    for(int n_n = 0; n_n < n_r; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        tmp_c[n->r] = n->c;
    }
    
    num_children = (int*) allocate_on_gpu(dev, n_r*sizeof(int));
    send_to_gpu(dev, tmp_c, num_children, n_r*sizeof(int));
    delete tmp_c;
}


HMF_AUGLAG_GPU_SOLVER_BASE::~HMF_AUGLAG_GPU_SOLVER_BASE(){
    deallocate_on_gpu(dev, ps_ind);
    deallocate_on_gpu(dev, num_children);
}

void HMF_AUGLAG_GPU_SOLVER_BASE::block_iter(){
    
    //calculate the capacity and then update flows
    calc_capacity_hmf(dev, g, ps_ind, div, pt, u_tmp, n_s, n_r, icc, tau/(float)(n_r-n_c+1));
    /*calc_capacity_potts_source_separate(dev, g, div, pt, u_tmp, n_s, n_r, icc, tau);
    for(int n_n = 0; n_n < n_r; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        int r = n->r;
        if( n->parent->parent == NULL )
            inc_mult_buffer(dev, ps, g+r*n_s, n_s, -tau);
        else
            inc_mult_buffer(dev, pt+n->parent->r*n_s, g+r*n_s, n_s, -tau);
    }*/
    update_spatial_flow_calc();
                 
	//std::cout << "Printing flows" << std::endl;
    //for(int n_n = 0; n_n < n_r+1; n_n++){
    //    const TreeNode* n = bottom_up_list[n_r-n_n];
    //    float* n_pt_buf = pt+n->r*n_s;
    //    if(n->r == -1)
	//		print_buffer(dev, ps, n_s);
	//	else
	//		print_buffer(dev, n_pt_buf, n_s);
	//}
	//std::cout << std::endl;
		
    //update source and sink multipliers top down
    //for(int i = 0; i < 2*(n_r-n_c)+1; i++){
    //    update_flow_hmf(dev, g_ind, g_ps, g, ps_ind, ps, pt, div, u_tmp, icc, num_children, bottom_up_list[n_r]->c, n_s, n_r);
    //
    //    //constrain leaf sink flows
    //    max_neg_constrain(dev, pt, data, n_s*n_c);
    //}
    
    /*
    //prepare buffers with self value
    set_buffer(dev, g_ps, icc, n_s);
    prep_flow_hmf(dev, g, ps_ind, pt, div, u_tmp, icc, n_s, n_r);
    compute_parents_flow_hmf(dev, g_ind, pt, div, u_tmp, icc, n_s, n_r);
    
    //divide out and store
    divide_out_and_store_hmf(dev, g_ps, g, ps, pt, num_children, bottom_up_list[n_r]->c, n_s, n_r);
    //mult_buffer(dev, 1.0f / (float) bottom_up_list[n_r]->c, g_ps, n_s);
    //for(int n_n = n_c; n_n < n_r; n_n++)
    //    mult_buffer(dev, 1.0f / (float) (bottom_up_list[n_n]->c+1), g+bottom_up_list[n_n]->r*n_s, n_s);
    
    //move from temporary into actual buffer
    //copy_buffer(dev, g_ps, ps, n_s);
    //copy_buffer(dev, g, pt, n_s*n_r);
    max_neg_constrain(dev, pt, data, n_s*n_c);
    */
    
    for(int n_n = 0; n_n < n_r+1; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        float* n_pt_buf = pt+n->r*n_s;
        float* n_g_buf = g+n->r*n_s;
        float* n_div_buf = div+n->r*n_s;
        float* n_u_buf = u_tmp+n->r*n_s;
        
        //completed all leaves so constrain them
        if(n_n == n_c)
            max_neg_constrain(dev, pt, data, n_s*n_c);
    
        //if we are the source node
        if(n->r == -1){
			//std::cout << "Source " << n->r << " " << n->d << std::endl;
            set_buffer(dev, ps, icc, n_s);
            for(int c = 0; c < n->c; c++){
                const TreeNode* nc = n->children[c];
                float* c_pt_buf = pt+nc->r*n_s;
                float* c_div_buf = div+nc->r*n_s;
                float* c_u_buf = u_tmp+nc->r*n_s;
				inc_inc_minc_buffer(dev, c_pt_buf, c_div_buf, c_u_buf, -icc, ps, n_s);
				//print_buffer(dev, c_pt_buf, n_s);
				//print_buffer(dev, c_div_buf, n_s);
				//print_buffer(dev, c_u_buf, n_s);
            }
            mult_buffer(dev, 1.0f / (float) n->c, ps, n_s);
			//print_buffer(dev, ps, n_s);
			//std::cout << std::endl;
        }

        //if we are a branch node
        else if(n->c > 0){
			//std::cout << "Branch " << n->r << " " << n->d << std::endl;
            const TreeNode* p = n->parent;
            float* p_pt_buf = pt+p->r*n_s;
            if( p->r == -1 )
                p_pt_buf = ps;
            copy_buffer(dev, p_pt_buf, n_pt_buf, n_s);
            ninc_buffer(dev, n_div_buf, n_pt_buf, n_s);
            inc_mult_buffer(dev, n_u_buf, n_pt_buf, n_s, icc);
            for(int c = 0; c < n->c; c++){
                const TreeNode* nc = n->children[c];
                float* c_pt_buf = pt+nc->r*n_s;
                float* c_div_buf = div+nc->r*n_s;
                float* c_u_buf = u_tmp+nc->r*n_s;
				//print_buffer(dev, c_pt_buf, n_s);
				//print_buffer(dev, c_div_buf, n_s);
				//print_buffer(dev, c_u_buf, n_s);
				inc_inc_minc_buffer(dev, c_pt_buf, c_div_buf, c_u_buf, -icc, n_pt_buf, n_s);
            }
            mult_buffer(dev, 1.0f / (float) (n->c+1), n_pt_buf, n_s);
			//print_buffer(dev, p_pt_buf, n_s);
			//print_buffer(dev, n_div_buf, n_s);
			//print_buffer(dev, n_u_buf, n_s);
			//print_buffer(dev, n_pt_buf, n_s);
			//std::cout << std::endl;
        }
    
        //if we are a leaf node
        else{
			//std::cout << "Leaf " << n->r << " " << n->d << std::endl;
            const TreeNode* p = n->parent;
            float* p_pt_buf = pt+p->r*n_s;
            if( p->r == -1 )
                p_pt_buf = ps;
            copy_buffer(dev, p_pt_buf, n_pt_buf, n_s);
            ninc_buffer(dev, n_div_buf, n_pt_buf, n_s);
            inc_mult_buffer(dev, n_u_buf, n_pt_buf, n_s, icc);
			//print_buffer(dev, p_pt_buf, n_s);
			//print_buffer(dev, n_div_buf, n_s);
			//print_buffer(dev, n_u_buf, n_s);
			//print_buffer(dev, n_pt_buf, n_s);
			//std::cout << std::endl;
    
        }
    }
    
    //update multipliers
    update_multiplier_hmf(dev, ps_ind, div, pt, u_tmp, g, n_s, n_r, cc);
    /*copy_buffer(dev, pt, g, n_s*n_r);
    inc_buffer(dev, div, g, n_s*n_r);
    for(int n_n = 0; n_n < n_r; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        const TreeNode* p = n->parent;
        float* n_g_buf = g+n->r*n_s;
        float* p_pt_buf = pt+p->r*n_s;
        if( p->r == -1 )
            p_pt_buf = ps;
        ninc_buffer(dev, p_pt_buf, n_g_buf, n_s);
    }
    mult_buffer(dev, -cc, g, n_s*n_r);
    inc_buffer(dev, g, u_tmp, n_s*n_r);
    */
                 
	//std::cout << "Printing flows" << std::endl;
    //for(int n_n = 0; n_n < n_r+1; n_n++){
    //    const TreeNode* n = bottom_up_list[n_r-n_n];
    //    float* n_pt_buf = pt+n->r*n_s;
    //    if(n->r == -1)
	//		print_buffer(dev, ps, n_s);
	//	else
	//		print_buffer(dev, n_pt_buf, n_s);
	//}
	//std::cout << std::endl;
}

void HMF_AUGLAG_GPU_SOLVER_BASE::operator()(){
	
    //initialize other variables
    clear_spatial_flows();
    clear_buffer(dev, div, n_s*n_r);
    find_min_constraint(dev, ps, data, n_c, n_s);
    for(int c = 0; c < n_r; c++)
        copy_buffer(dev, ps, pt+c*n_s, n_s);
    
    //initialize labels
	clear_buffer(dev, u_tmp+n_s*n_c, n_s*(n_r-n_c));
    mark_neg_equal(dev, ps, data, u_tmp, n_s, n_c);
    for (int l = n_c; l < n_r; l++) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(dev, u_tmp+n->children[c]->r*n_s, u_tmp+n->r*n_s, n_s);
    }
    
    // iterate in blocks
    int min_iter = min_iter_calc();
    if( min_iter < 10 )
        min_iter = 10;
    int max_loop = 200;

    for(int i = 0; i < max_loop; i++){    
        
        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

        //Determine if converged
        float max_change = max_of_buffer(dev, g, n_s*n_r);
		//std::cout << "Iter " << i << ": " << max_change << std::endl;
        if (max_change < beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();
    
    //get final output
    //copy_buffer(dev, u_tmp, u, n_s*n_c);
    log_buffer(dev, u_tmp, u, n_s*n_c);

}

#endif
