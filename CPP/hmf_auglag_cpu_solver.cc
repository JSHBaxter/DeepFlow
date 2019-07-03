#include "hmf_auglag_cpu_solver.h"
#include <math.h>
#include <thread>
#include <iostream>
#include <limits>
#include "cpu_kernels.h"
#include "hmf_trees.h"

HMF_AUGLAG_CPU_SOLVER_BASE::HMF_AUGLAG_CPU_SOLVER_BASE(
    TreeNode** bottom_up_list,
    const int batch,
    const int n_s,
    const int n_c,
    const int n_r,
    const float* data_cost,
    float* u ) :
bottom_up_list(bottom_up_list),
b(batch),
n_c(n_c),
n_r(n_r),
n_s(n_s),
data(data_cost),
ps(0),
u_tmp(0),
pt(0),
div(0),
g(0),
data_b(0),
u(u)
{std::cout << n_s << " " << n_c << " " << n_r << std::endl;}

//perform one iteration of the algorithm
void HMF_AUGLAG_CPU_SOLVER_BASE::block_iter(){
    
    //calculate the capacity and then update flows
    for(int n_n = 0; n_n < n_r; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        int r = n->r;
        if( n->parent->parent == NULL )
            compute_capacity_potts(g+r*n_s, u_tmp+r*n_s, ps, pt+r*n_s, div+r*n_s, n_s, 1, tau, icc);
        else
            compute_capacity_potts(g+r*n_s, u_tmp+r*n_s, pt+n->parent->r*n_s, pt+r*n_s, div+r*n_s, n_s, 1, tau, icc);
    }
    update_spatial_flow_calc();
                 
	//std::cout << "Printing flows" << std::endl;
    //for(int n_n = 0; n_n < n_r+1; n_n++){
    //    const TreeNode* n = bottom_up_list[n_r-n_n];
    //    float* n_pt_buf = pt+n->r*n_s;
    //    if(n->r == -1)
	//		print_buffer(ps, n_s);
	//	else
	//		print_buffer(n_pt_buf, n_s);
	//}
	//std::cout << std::endl;

    //update source and sink multipliers top down
    for(int n_n = 0; n_n < n_r+1; n_n++){
        const TreeNode* n = bottom_up_list[n_r-n_n];
        float* n_pt_buf = pt+n->r*n_s;
        float* n_g_buf = g+n->r*n_s;
        float* n_div_buf = div+n->r*n_s;
        float* n_u_buf = u_tmp+n->r*n_s;

        //if we are the source node
        if(n->r == -1){
			//std::cout << "Source " << n->r << " " << n->d << std::endl;
            set(ps, icc, n_s);
            for(int c = 0; c < n->c; c++){
                const TreeNode* nc = n->children[c];
                float* c_pt_buf = pt+nc->r*n_s;
                float* c_div_buf = div+nc->r*n_s;
                float* c_u_buf = u_tmp+nc->r*n_s;
                inc(c_pt_buf, ps, n_s);
                inc(c_div_buf, ps, n_s);
                inc(c_u_buf, ps, -icc, n_s);
				//print_buffer(c_pt_buf, n_s);
				//print_buffer(c_div_buf, n_s);
				//print_buffer(c_u_buf, n_s);
            }
            mult_buffer(ps, 1.0f / (float) n->c, n_s);
			//print_buffer(ps, n_s);
			//std::cout << std::endl;
        }

        //if we are a branch node
        else if(n->c > 0){
			//std::cout << "Branch " << n->r << " " << n->d << std::endl;
            const TreeNode* p = n->parent;
            float* p_pt_buf = pt+p->r*n_s;
            if( p->r == -1 )
                p_pt_buf = ps;
            copy(p_pt_buf,n_pt_buf,n_s);
            ninc(n_div_buf, n_pt_buf, n_s);
            inc(n_u_buf, n_pt_buf, icc, n_s);
            for(int c = 0; c < n->c; c++){
                const TreeNode* nc = n->children[c];
                float* c_pt_buf = pt+nc->r*n_s;
                float* c_div_buf = div+nc->r*n_s;
                float* c_u_buf = u_tmp+nc->r*n_s;
                inc(c_pt_buf, n_pt_buf, n_s);
                inc(c_div_buf, n_pt_buf, n_s);
                inc(c_u_buf, n_pt_buf, -icc, n_s);
				//print_buffer(c_pt_buf, n_s);
				//print_buffer(c_div_buf, n_s);
				//print_buffer(c_u_buf, n_s);
            }
            mult_buffer(n_pt_buf, 1.0f / (float) (n->c+1), n_s);
			//print_buffer(p_pt_buf, n_s);
			//print_buffer(n_div_buf, n_s);
			//print_buffer(n_u_buf, n_s);
			//print_buffer(n_pt_buf, n_s);
			//std::cout << std::endl;
        }

        //if we are a leaf node
        else{
			//std::cout << "Leaf " << n->r << " " << n->d << std::endl;
            const float* n_d_buf = data_b+n->r*n_s;
            const TreeNode* p = n->parent;
            float* p_pt_buf = pt+p->r*n_s;
            if( p->r == -1 )
                p_pt_buf = ps;
            copy(p_pt_buf,n_pt_buf,n_s);
            ninc(n_div_buf, n_pt_buf, n_s);
            inc(n_u_buf, n_pt_buf, icc, n_s);
			//print_buffer(p_pt_buf, n_s);
			//print_buffer(n_div_buf, n_s);
			//print_buffer(n_u_buf, n_s);
			//print_buffer(n_pt_buf, n_s);
			//std::cout << std::endl;
            constrain(n_pt_buf,n_d_buf,n_s);
        }
    }
    
    //update multipliers
    copy(pt, g, n_s*n_r);
    inc(div, g, n_s*n_r);
    for(int n_n = 0; n_n < n_r; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        const TreeNode* p = n->parent;
        float* n_g_buf = g+n->r*n_s;
        float* p_pt_buf = pt+p->r*n_s;
        if( p->r == -1 )
            p_pt_buf = ps;
        ninc(p_pt_buf,n_g_buf,n_s);
    }
    mult_buffer(g, -cc, n_s*n_r);
    inc(g, u_tmp, n_s*n_r);
                 
	//std::cout << "Printing flows" << std::endl;
    //for(int n_n = 0; n_n < n_r+1; n_n++){
    //    const TreeNode* n = bottom_up_list[n_r-n_n];
    //    float* n_pt_buf = pt+n->r*n_s;
    //    if(n->r == -1)
	//		print_buffer(ps, n_s);
	//	else
	//		print_buffer(n_pt_buf, n_s);
	//}
	//std::cout << std::endl;
}

void HMF_AUGLAG_CPU_SOLVER_BASE::operator()(){

    //store intermediate information
    ps = new float[n_s];
    u_tmp = new float[n_s*n_r];
    pt = new float[n_s*n_r];
    div = new float[n_s*n_r];
    g = new float[n_s*n_r];
    data_b = new float[n_s*n_r];

    // transpose input data (makes everything easier)
    for(int s = 0; s < n_s; s++)
        for(int c = 0; c < n_c; c++)
            data_b[c*n_s+s] = -data[s*n_c+c];

    //initialize variables
    clear(g, div, u_tmp, n_r*n_s);
    clear_spatial_flows();
    clear(pt, n_r*n_s);
    clear(ps, n_s);
    init_flows_channels_first(data_b, ps, n_c, n_s);
    for(int i = 0; i < n_r; i++)
        copy(ps,pt+i*n_s,n_s);

    // iterate in blocks
    int min_iter = min_iter_calc();
    if (min_iter < 10)
        min_iter = 10;
    int max_loop = 200;
    for(int i = 0; i < max_loop; i++){

        //run the solver a set block of iterations
        for (int iter = 0; iter < min_iter; iter++)
            block_iter();

        float max_change = maxabs(g,n_s*n_r);
		std::cout << "Iter " << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();

    //log output and transpose output back into proper buffer
    //log_buffer(u_tmp, n_s*n_c);
    for(int s = 0; s < n_s; s++)
        for(int c = 0; c < n_c; c++)
            u[s*n_c+c] = u_tmp[c*n_s+s];
        
    //deallocate temporary buffers
    delete u_tmp; u_tmp = 0;
    delete pt; pt = 0;
    delete ps; ps = 0;
    delete g; g = 0;
    delete div; div = 0;
    delete data_b; data_b = 0;
    clean_up();
}

HMF_AUGLAG_CPU_SOLVER_BASE::~HMF_AUGLAG_CPU_SOLVER_BASE(){
}