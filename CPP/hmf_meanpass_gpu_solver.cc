#include "hmf_meanpass_gpu_solver.h"
#include "gpu_kernels.h"

HMF_MEANPASS_GPU_SOLVER_BASE::HMF_MEANPASS_GPU_SOLVER_BASE(
    const GPUDevice & dev,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_s,
    const int n_c,
    const int n_r,
    const float* const data_cost,
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
temp(full_buff[0])
{}
    
void HMF_MEANPASS_GPU_SOLVER_BASE::block_iter(){
    
    //calculate the aggregate probabilities (stored in temp)
    copy_buffer(dev, u, temp, n_s*n_c);
    clear_buffer(dev, temp+n_c*n_s, n_s*(n_r-n_c));
    for (int l = n_c; l < n_r; l++) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(dev, temp+n->children[c]->r*n_s, temp+n->r*n_s, n_s);
    }
	//print_buffer(dev, temp, n_s*n_r);

    //calculate the effective regularization (overwrites own temp)
    update_spatial_flow_calc();
	//print_buffer(dev, temp, n_s*n_r);

    //calculate the aggregate effective regularization (overwrites own temp)
    for (int l = n_r-1; l >= n_c; l--) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(dev, temp+n->r*n_s, temp+n->children[c]->r*n_s, n_s);
    }
	//print_buffer(dev, temp, n_s*n_r);

    // get new probability estimates, and normalize (store answer in temp)
    softmax(dev, data, temp, temp, n_s, n_c);

    //update labels 
    change_to_diff(dev, u, temp, n_s*n_c, tau);
}

void HMF_MEANPASS_GPU_SOLVER_BASE::operator()(){

    //initialize variables
    softmax(dev, data, NULL, u, n_s, n_c);

    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = 200;
    
    for(int i = 0; i < max_loop; i++){    
        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

        //Determine if converged
        float max_change = max_of_buffer(dev, temp, n_s*n_c);
        std::cout << "Iter #" << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }
    
    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();

    //calculate the aggregate probabilities
    copy_buffer(dev, u, temp, n_s*n_c);
    clear_buffer(dev, temp+n_c*n_s, n_s*(n_r-n_c));
    for (int l = n_c; l < n_r; l++) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(dev, temp+n->children[c]->r*n_s, temp+n->r*n_s, n_s);
    }

    //calculate the effective regularization
    update_spatial_flow_calc();

    //calculate the aggregate effective regularization
    for (int l = n_r-1; l >= n_c; l--) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(dev, temp+n->r*n_s, temp+n->children[c]->r*n_s, n_s);
    }

    
    //get final output
    add_then_store(dev, data, temp, u, n_c*n_s);
    
}

HMF_MEANPASS_GPU_GRADIENT_BASE::HMF_MEANPASS_GPU_GRADIENT_BASE(
    const GPUDevice & dev,
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
u(u),
grad(g),
u_tmp(full_buff[0]),
dy(full_buff[1]),
du(full_buff[2]),
tmp(full_buff[3])
{}

void HMF_MEANPASS_GPU_GRADIENT_BASE::operator()(){
    
    int min_iter = min_iter_calc();
    if( min_iter < 10 )
        min_iter = 10;
    int max_loop = 200;
    
    //calculate the aggregate probabilities
    softmax(dev, u, NULL, u_tmp, n_s, n_c);
    clear_buffer(dev, u_tmp+n_c*n_s, n_s*(n_r-n_c));
    for (int l = n_c; l < n_r; l++) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(dev, u_tmp+n->children[c]->r*n_s, u_tmp+n->r*n_s, n_s);
    }
    
    //calculate aggregate gradient
    copy_buffer_clip(dev, grad, dy, n_s*n_c, 1.0f/(n_s*n_c));
    clear_buffer(dev, dy+n_s*n_c, n_s*(n_r-n_c));
    for (int l = n_c; l < n_r; l++) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(dev, dy+n->children[c]->r*n_s, dy+n->r*n_s, n_s);
    }
  
    // populate data gradient
    copy_buffer(dev, dy, g_data, n_s*n_c);
    
    //and calculate gradients for the rest
    update_spatial_flow_calc();

    //collapse back down to leaves
    for (int l = n_r-1; l >= n_c; l--) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc_buffer(dev, du+n->r*n_s, du+n->children[c]->r*n_s, n_s);
    }
    
    for(int i = 0; i < max_loop; i++){
        //push gradients back a number of iterations (first iteration has tau=1, the rest a smaller tau)
        for(int iter = 0; iter < min_iter; iter++){

            //process gradients and expand them upwards
            process_grad_potts(dev, du, u_tmp, dy, n_s, n_c, tau);
            clear_buffer(dev, dy+n_s*n_c, n_s*(n_r-n_c));
            for (int l = n_c; l < n_r; l++) {
                const TreeNode* n = bottom_up_list[l];
                for(int c = 0; c < n->c; c++)
                    inc_buffer(dev, dy+n->children[c]->r*n_s, dy+n->r*n_s, n_s);
            }

            //add into data term gradient
            inc_buffer(dev, dy, g_data, n_s*n_c);

            //get gradients for the regularization terms
            update_spatial_flow_calc();

            //collapse back down to leaves
            for (int l = n_r-1; l >= n_c; l--) {
                const TreeNode* n = bottom_up_list[l];
                for(int c = 0; c < n->c; c++)
                    inc_buffer(dev, tmp+n->r*n_s, tmp+n->children[c]->r*n_s, n_s);
            }

            //add diminished content from before
            mult_buffer(dev, 1.0f-tau-epsilon, du, n_s*n_c);
            inc_buffer(dev, tmp, du, n_c*n_s);
        }
        
        copy_buffer(dev, du, dy, n_s*n_c);
        float max_change = max_of_buffer(dev, dy, n_c*n_s);
        if(max_change < beta)
            break;
    }
    
}