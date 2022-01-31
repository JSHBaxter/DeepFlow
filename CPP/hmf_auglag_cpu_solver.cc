#include "hmf_auglag_cpu_solver.h"
#include <math.h>
#include <thread>
#include <iostream>
#include <limits>
#include "cpu_kernels.h"
#include "hmf_trees.h"

HMF_AUGLAG_CPU_SOLVER_BASE::~HMF_AUGLAG_CPU_SOLVER_BASE(){
    delete ps;
}

HMF_AUGLAG_CPU_SOLVER_BASE::HMF_AUGLAG_CPU_SOLVER_BASE(
    const bool channels_first,
    TreeNode** bottom_up_list,
    const int batch,
    const int n_s,
    const int n_c,
    const int n_r,
    const float* data_cost,
    float* u ) :
channels_first(channels_first),
bottom_up_list(bottom_up_list),
b(batch),
n_c(n_c),
n_r(n_r),
n_s(n_s),
data(data_cost),
u(u),
ps(new float[n_s+4*n_r*n_s+n_c*n_s]),
u_tmp(ps+n_s),
pt(u_tmp+n_s*n_r),
div(pt+n_s*n_r),
g(div+n_s*n_r),
data_b(g+n_s*n_r)
{
    // invert data term from a positive benefit to a negative cost (to save on mult's later)
    if(channels_first)
        copy(data,data_b,n_c*n_s);
    else
        transpose(data,data_b,n_s,n_c);
    mult_buffer(data_b,-1.0f,n_s*n_c);
}

//perform one iteration of the algorithm
void HMF_AUGLAG_CPU_SOLVER_BASE::block_iter(){
    
    //calculate the capacity and then update flows (note we don't need to cover the source node)
    for(int n_n = 0; n_n < n_r; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
	    //std::cout << "TreeNode ptr: " << n << std::endl;
        int r = n->r;
        if( n->parent->parent == NULL ){
    	    //std::cout << "Calc capacities" << g+r*n_s << " " << u_tmp+r*n_s << " " << ps << " " << pt+r*n_s << " " << div+r*n_s << " " << std::endl;
            compute_capacity_binary(g+r*n_s, u_tmp+r*n_s, ps, pt+r*n_s, div+r*n_s, n_s, tau, icc);
        }else{
    	    //std::cout << "Calc capacities" << g+r*n_s << " " << u_tmp+r*n_s << " " << pt+n->parent->r*n_s << " " << pt+r*n_s << " " << div+r*n_s << " " << std::endl;
            compute_capacity_binary(g+r*n_s, u_tmp+r*n_s, pt+n->parent->r*n_s, pt+r*n_s, div+r*n_s, n_s, tau, icc);
        }
    }
    update_spatial_flow_calc();

    //update source and sink multipliers bottom up
    for(int n_n = 0; n_n < n_r+1; n_n++){
        const TreeNode* n = bottom_up_list[n_n];
        float* n_pt_buf = pt+n->r*n_s;
        float* n_g_buf = g+n->r*n_s;
        float* n_div_buf = div+n->r*n_s;
        float* n_u_buf = u_tmp+n->r*n_s;

        //if we are finished the leaves,
        if(n_n == n_c)
            constrain(pt+n_s*(n_r-n_c),data_b,n_s*n_c);
        
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
    int u_tmp_offset = n_s*(n_r-n_c);

    //initialize flows
    clear(g, div, n_r*n_s);
    clear_spatial_flows();
    clear(pt, n_r*n_s);
    clear(ps, n_s);
    init_flows_channels_first(data_b, ps, n_c, n_s);
    for(int i = 0; i < n_r; i++)
        copy(ps,pt+i*n_s,n_s);

    //initialize multipliers
    clear(u_tmp, n_s*(n_r-n_c));
    for(int i = 0, c = 0; c < n_c; c++)
        for (int s = 0; s < n_s; s++, i++)
            if (ps[s] == data_b[i])
                u_tmp[u_tmp_offset+i] = 1.0f;
            else
                u_tmp[u_tmp_offset+i] = 0.0f;
    for (int l = n_c; l < n_r; l++) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
            inc(u_tmp+n->children[c]->r*n_s, u_tmp+n->r*n_s, n_s);
    }

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
	    //std::cout << "HMF_AUGLAG_CPU_SOLVER_BASE Iter " << i << ": " << max_change << std::endl;
        if (max_change < tau*beta)
            break;
    }

    //run one last block, just to be safe
    for (int iter = 0; iter < min_iter; iter++)
        block_iter();

    //log output and transpose output back into proper buffer
    log_buffer(u_tmp+u_tmp_offset,n_s*n_c);
    if( !channels_first ){
        for(int s = 0; s < n_s; s++)
            for(int c = 0; c < n_c; c++)
                u[s*n_c+c] = u_tmp[u_tmp_offset+c*n_s+s];
    }else{
        copy(u_tmp+u_tmp_offset,u,n_s*n_c);
    }
        
    //finishing tasks
    clean_up();
}

