
#include "hmf_meanpass_gpu_solver.h"
#include "gpu_kernels.h"
#include <iostream>

HMF_MEANPASS_GPU_SOLVER_BASE::HMF_MEANPASS_GPU_SOLVER_BASE(
    const cudaStream_t & dev,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_s,
    const int n_c,
    const int n_r,
    const float* const data_cost,
	const float* const init_u,
    float* const u,
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
r_eff(full_buff[0]),
u_full(full_buff[1])
{
	std::cout << n_s << " " << n_c << " " << n_r << std::endl;
	//initialize variables
	if(init_u)
		copy_buffer(dev, init_u, u, n_s*n_c);
	else
		softmax(dev, data, NULL, u, n_s, n_c);
    
    //get pointers to parents' u buffer
    float** tmp_u_ind = new float* [n_r];
    for(int n_n = 0; n_n < n_r; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        if(n->parent->parent == NULL)
            tmp_u_ind[n->r] = 0;
        else
            tmp_u_ind[n->r] = u_full + n_s*n->parent->r;
    }
    u_ind = (float**) allocate_on_gpu(dev, 2*n_r*sizeof(float*));
    send_to_gpu(dev, tmp_u_ind, u_ind, n_r*sizeof(float*));
    delete tmp_u_ind;
    
    //get pointers to parents' regularization buffer
    float** tmp_r_ind = new float* [n_r];
    for(int n_n = 0; n_n < n_r; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        if(n->parent->parent == NULL)
            tmp_r_ind[n->r] = 0;
        else
            tmp_r_ind[n->r] = r_eff + n_s*n->parent->r;
    }
    reg_ind = u_ind+n_r;
    send_to_gpu(dev, tmp_r_ind, reg_ind, n_r*sizeof(float*));
    delete tmp_r_ind;
}

HMF_MEANPASS_GPU_SOLVER_BASE::~HMF_MEANPASS_GPU_SOLVER_BASE(){
    deallocate_on_gpu(dev, u_ind);
}
void HMF_MEANPASS_GPU_SOLVER_BASE::block_iter(const int parity){
    
    //calculate the aggregate probabilities (stored in u_full)
    copy_buffer(dev, u, u_full+n_s*(n_r-n_c), n_s*n_c);
    clear_buffer(dev, u_full, n_s*(n_r-n_c));
    for (int l = n_c; l < n_r; l++) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(dev, u_full+n->children[c]->r*n_s, u_full+n->r*n_s, n_s);
    }
	//print_buffer(dev, r_eff, n_s*n_r);
    //aggregate_bottom_up(dev, u_ind, u_full, u, n_s, n_c, n_r);

    //calculate the effective regularization (overwrites own r_eff)
    update_spatial_flow_calc();
	//print_buffer(dev, r_eff, n_s*n_r);

    //calculate the aggregate effective regularization (overwrites own r_eff)
    for (int l = n_r-1; l >= 0; l--) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(dev, r_eff+n->r*n_s, r_eff+n->children[c]->r*n_s, n_s);
    }
	//print_buffer(dev, r_eff, n_s*n_r);

    // get new probability estimates, and normalize
    // (copy into u_full as tmp space (prevent race condition) then store answer in r_eff)
    copy_buffer(dev, r_eff+n_s*(n_r-n_c), u_full, n_s*n_c);
    softmax(dev, data, u_full, r_eff, n_s, n_c);

    //update labels
    parity_merge_buffer(r_eff,u,parity);
    change_to_diff(dev, u, r_eff, n_s*n_c, tau);
}

void HMF_MEANPASS_GPU_SOLVER_BASE::operator()(){
    
    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = 200;
    
    for(int i = 0; i < max_loop; i++){    
        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter-1; iter++){
            block_iter(0);
            block_iter(1);
        }

        //Determine if converged
        block_iter(0);
        float max_change_1 = max_of_buffer(dev, r_eff, n_s*n_c);
        block_iter(1);
        float max_change_2 = max_of_buffer(dev, r_eff, n_s*n_c);
        float max_change = (max_change_1 > max_change_2) ? max_change_1 : max_change_2;
        //std::cout << "HMF_MEANPASS_GPU_SOLVER_BASE Iter #" << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }
    
    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++){
        block_iter(0);
        block_iter(1);
    }

    //calculate the aggregate probabilities
    copy_buffer(dev, u, u_full+n_s*(n_r-n_c), n_s*n_c);
    clear_buffer(dev, u_full, n_s*(n_r-n_c));
    for (int l = n_c; l < n_r; l++) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(dev, u_full+n->children[c]->r*n_s, u_full+n->r*n_s, n_s);
    }

    //calculate the effective regularization
    update_spatial_flow_calc();

    //calculate the aggregate effective regularization
    for (int l = n_r-1; l >= 0; l--) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(dev, r_eff+n->r*n_s, r_eff+n->children[c]->r*n_s, n_s);
    }

    
    //get final output
    add_then_store(dev, data, r_eff+n_s*(n_r-n_c), u, n_c*n_s);
    
}

HMF_MEANPASS_GPU_GRADIENT_BASE::HMF_MEANPASS_GPU_GRADIENT_BASE(
    const cudaStream_t & dev,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_s,
    const int n_c,
    const int n_r,
    const float* const u,
    const float* const g,
    float* const g_d,
    float** full_buff) :
dev(dev),
bottom_up_list(bottom_up_list),
b(batch),
n_s(n_s),
n_c(n_c),
n_r(n_r),
g_data(g_d),
logits(u),
grad(g),
u(full_buff[0]),
dy(full_buff[1]),
du(full_buff[2]),
tmp(full_buff[3])
{}

void HMF_MEANPASS_GPU_GRADIENT_BASE::block_iter(){

		//process gradients 
		untangle_softmax(dev, du+n_s*(n_r-n_c), u+n_s*(n_r-n_c), dy+n_s*(n_r-n_c), n_s, n_c);

		//add into data term gradient
		inc_mult_buffer(dev, dy+n_s*(n_r-n_c), g_data, n_s*n_c, tau);

        //expand gradients upwards
		clear_buffer(dev, dy, n_s*(n_r-n_c));
		for (int l = n_c; l < n_r; l++) {
			const TreeNode* n = bottom_up_list[l];
			for(int c = 0; c < n->c; c++)
				inc_buffer(dev, dy+n->children[c]->r*n_s, dy+n->r*n_s, n_s);
		}
    
		//get gradients for the regularization terms
		get_reg_gradients_and_push(tau);

		//collapse back down to leaves
		for (int l = n_r-1; l >= n_c; l--) {
			const TreeNode* n = bottom_up_list[l];
			for(int c = 0; c < n->c; c++)
				inc_buffer(dev, du+n->r*n_s, du+n->children[c]->r*n_s, n_s);
		}
}

void HMF_MEANPASS_GPU_GRADIENT_BASE::operator()(){
    int u_tmp_offset = n_s*(n_r-n_c);
    
    //get initial maximum value of teh gradient for convgencence purposes
    const float init_grad_max = max_of_buffer(dev, grad, n_s*n_c);
    
	clear_variables();
    int min_iter = min_iter_calc();
    if( min_iter < 10 )
        min_iter = 10;
    int max_loop = 200;
    
    //calculate the aggregate probabilities
    clear_buffer(dev, u, n_s*(n_r-n_c));
    softmax(dev, logits, NULL, u+u_tmp_offset, n_s, n_c);
    for (int l = n_c; l < n_r; l++) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(dev, u+n->children[c]->r*n_s, u+n->r*n_s, n_s);
    }

    // populate data gradient
    copy_buffer(dev, grad, g_data, n_s*n_c);
    
    //calculate aggregate gradient
    copy_buffer(dev, grad, dy+n_s*(n_r-n_c), n_s*n_c);
    clear_buffer(dev, dy, n_s*(n_r-n_c));
    for (int l = n_c; l < n_r; l++) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(dev, dy+n->children[c]->r*n_s, dy+n->r*n_s, n_s);
    }
    
    //and calculate gradients for the rest
    clear_buffer(dev,du,n_s*n_r);
    get_reg_gradients_and_push(1.0f);

    //collapse back down to leaves
    for (int l = n_r-1; l >= n_c; l--) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(dev, du+n->r*n_s, du+n->children[c]->r*n_s, n_s);
    }
	
    for(int i = 0; i < max_loop; i++){
		//push gradients back a number of iterations (first iteration has tau=1, the rest a smaller tau)
		for(int iter = 0; iter < min_iter; iter++)
			block_iter();
        
        float max_change = max_of_buffer(dev, du+u_tmp_offset, n_c*n_s);
        //std::cout << "HMF_MEANPASS_GPU_GRADIENT_BASE Iter " << i << " " << max_change << std::endl;
        if(max_change < beta*init_grad_max)
            break;
    }
}
