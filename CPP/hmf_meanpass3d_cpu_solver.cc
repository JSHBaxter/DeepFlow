#include <math.h>
#include <thread>
#include <iostream>
#include <limits>
#include "cpu_kernels.h"
#include "hmf_trees.h"


namespace HMF_CPU {
    
class SolverBatchThreadChannelsLast
{
private:
    TreeNode const* const* bottom_up_list;
    const int b;
    const int n_x;
    const int n_y;
    const int n_z;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* data;
    const float* rx;
    const float* ry;
    const float* rz;
    float* u;
        
	// optimization constants
	const float tau = 0.5f;
	const float beta = 0.02f;
	const float epsilon = 10e-5f;
    
public:
    SolverBatchThreadChannelsLast(
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[7],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
        float* u ) :
    bottom_up_list(bottom_up_list),
    b(batch),
    n_x(sizes[1]),
    n_y(sizes[2]),
    n_z(sizes[3]),
    n_c(sizes[4]),
    n_r(sizes[6]),
    n_s(sizes[1]*sizes[2]*sizes[3]),
    data(data_cost),
    rx(rx_cost),
    ry(ry_cost),
    rz(rz_cost),
    u(u)
    {}

    
    void operator()(){
        
        // allocate intermediate variables
        float max_change = 0.0f;
        float* u_tmp = new float[n_s*n_r];
        float* r_eff = new float[n_s*n_r];

        //initialize variables
        softmax(data, u, n_s, n_c);
        
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
            for (int iter = 0; iter < min_iter; iter++){
                aggregate_bottom_up(u,u_tmp,n_s,n_c,n_r,bottom_up_list);
                calculate_r_eff(r_eff, rx, ry, rz, u_tmp, n_x, n_y, n_z, n_r);
                aggregate_top_down(r_eff, n_s, n_r, bottom_up_list);
                for(int s = 0, i = 0; s < n_s; s++)
				for(int c = 0; c < n_c; c++, i++)
                    r_eff[i] = data[i]+r_eff[n_r*s+c];
                max_change = softmax_with_convergence(r_eff, u, n_s, n_c, tau);
            }

            if (max_change < beta)
                break;
        }

        //run one last block, just to be safe
        for (int iter = 0; iter < min_iter; iter++){
            aggregate_bottom_up(u,u_tmp,n_s,n_c,n_r,bottom_up_list);
            calculate_r_eff(r_eff, rx, ry, rz, u_tmp, n_x, n_y, n_z, n_r);
			aggregate_top_down(r_eff, n_s, n_r, bottom_up_list);
            for(int s = 0, i = 0; s < n_s; s++)
			for(int c = 0; c < n_c; c++, i++)
				r_eff[i] = data[i]+r_eff[n_r*s+c];
            max_change = softmax_with_convergence(r_eff, u, n_s, n_c, tau);
        }
        
        //perform majority of last iteration
        aggregate_bottom_up(u,u_tmp,n_s,n_c,n_r,bottom_up_list);
        calculate_r_eff(r_eff, rx, ry, rz, u_tmp, n_x, n_y, n_z, n_r);
		aggregate_top_down(r_eff, n_s, n_r, bottom_up_list);

        //get final output
        for(int s = 0; s < n_s; s++)
            for(int c = 0; c < n_c; c++)
                u[n_c*s+c] = data[n_c*s+c]+r_eff[n_r*s+c];
        
        //deallocate temporary buffers
        free(u_tmp);
        free(r_eff);
    
    }
};


class GradientBatchThreadChannelsLast
{
private:
    TreeNode const* const* bottom_up_list;
    const int b;
    const int n_x;
    const int n_y;
    const int n_z;
    const int n_c;
    const int n_r;
    const int n_s;
    float* g_data;
    float* g_rx;
    float* g_ry;
    float* g_rz;
    const float* logits;
    const float* grad;
	const float* rx;
	const float* ry;
	const float* rz;
	
	const float epsilon = 0.00001;
	const float beta = 1e-20;
	const float tau = 0.5;
    
public:
    GradientBatchThreadChannelsLast(
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[7],
        const float* u,
        const float* g,
        float* g_d,
		const float* rx_cost,
		const float* ry_cost,
		const float* rz_cost,
        float* g_rx,
        float* g_ry,
        float* g_rz ) :
    bottom_up_list(bottom_up_list),
    b(batch),
    n_x(sizes[1]),
    n_y(sizes[2]),
    n_z(sizes[3]),
    n_c(sizes[4]),
    n_r(sizes[6]),
    n_s(sizes[1]*sizes[2]*sizes[3]),
    g_data(g_d),
    g_rx(g_rx),
    g_ry(g_ry),
    g_rz(g_rz),
	rx(rx_cost),
	ry(ry_cost),
	rz(rz_cost),
    logits(u),
    grad(g)
    {}
    
    void operator()(){

        int max_loops = n_x;
        const int min_iters = 10;
		
        //allocate temporary variables
        float* u = new float[n_s*n_r];
        float* u_small = new float[n_s*n_c];
        float* dy = new float[n_s*n_r];
        float* g_u = new float[n_s*n_r];
        
        //transformat logits into labelling and unroll up through hierarchy
		clear(u, n_s*n_r);
        softmax(logits, u_small, n_s, n_c);
		copy(u_small, u, n_s*n_c);
        unfold_buffer(u, n_s, n_c, n_r);
		aggregate_bottom_up(u, n_s, n_r, bottom_up_list);

        //get initial gradient for the data terms and regularization terms
        copy(grad, g_data, n_s*n_c);
        copy(grad, dy, n_s*n_c);
        unfold_buffer(dy, n_s, n_c, n_r);
        get_reg_gradients(dy, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_r, 1.0f);
         
        //push gradient through energy equation
        get_gradient_for_u(dy, rx, ry, rz, g_u, n_x, n_y, n_z, n_r, 1);
        
		//collapse down to leaves
		aggregate_top_down(g_u, n_s, n_r, bottom_up_list);
		refold_buffer(g_u, n_s, n_c, n_r);
		
        for(int i = 0; i < max_loops; i++){
            for(int iter = 0; iter < min_iters; iter++){
                //untangle softmax
                untangle_softmax(g_u, u_small, dy, n_s, n_c);
                
                // populate data gradient
                inc(dy, g_data, tau, n_s*n_c);

                // unfold gradient to full hierarchy
				unfold_buffer(dy, n_s, n_c, n_r);
				
				//get gradient for regularization
                get_reg_gradients(dy, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_r, tau);

                //push gradient through energy equation
				get_gradient_for_u(dy, rx, ry, rz, g_u, n_x, n_y, n_z, n_r, tau);
        
				//collapse down to leaves
				aggregate_top_down(g_u, n_s, n_r, bottom_up_list);
				refold_buffer(g_u, n_s, n_c, n_r);
            }
            
            //get max of gu and break if converged
            float gu_max = maxabs(g_u, n_s*n_c);
            if( gu_max < beta )
                break;
        }
        
        delete u;
        delete u_small;
        delete dy;
        delete g_u;
        
    }
};


}

template <>
struct HmfMeanpass3dFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[7],
      const int* parentage,
      const int* data_index,
      const float* data_cost,
      const float* rx_cost,
      const float* ry_cost,
      const float* rz_cost,
      float* u,
      float** /*unused full buffers*/,
      float** /*unused image buffers*/){
      
    //build the tree
    TreeNode* node = NULL;
    TreeNode** children = NULL;
    TreeNode** bottom_up_list = NULL;
    TreeNode** top_down_list = NULL;
    TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, data_index, sizes[6], sizes[4]);
    //node->print_tree();
    //TreeNode::print_list(bottom_up_list, sizes[6]+1);
    //std::cout << "Tree built" << std::endl;

    int n_batches = sizes[0];
	int n_s = sizes[1]*sizes[2]*sizes[3];
	int n_c = sizes[4];
	int n_r = sizes[6];
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(HMF_CPU::SolverBatchThreadChannelsLast(bottom_up_list, b, sizes, data_cost + b*n_s*n_c,
																																													 rx_cost + b*n_s*n_r,
																																													 ry_cost + b*n_s*n_r,
																																													 rz_cost + b*n_s*n_r,
																																													 u + b*n_s*n_c));
    for(int b = 0; b < n_batches; b++)
        threads[b]->join();
    for(int b = 0; b < n_batches; b++)
        delete threads[b];
    delete threads;
      
    TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
      
  }
  int num_buffers_full(){ return 0; }
  int num_buffers_branch(){ return 0; }
  int num_buffers_data(){ return 0; }
  int num_buffers_images(){ return 0; }
};

template <>
struct HmfMeanpass3dGradFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[7],
      const int* parentage,
      const int* data_index,
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
      float** /*unused full buffers*/,
      float** /*unused image buffers*/){
      

    //build the tree
    TreeNode* node = NULL;
    TreeNode** children = NULL;
    TreeNode** bottom_up_list = NULL;
    TreeNode** top_down_list = NULL;
    TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, data_index, sizes[6], sizes[4]);
    //node->print_tree();
    //print_list(bottom_up_list, sizes[6]+1);
    //std::cout << "Tree built" << std::endl;
      
    int n_batches = sizes[0];
	int n_s = sizes[1]*sizes[2]*sizes[3];
	int n_c = sizes[4];
	int n_r = sizes[6];
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(HMF_CPU::GradientBatchThreadChannelsLast(bottom_up_list, b, sizes, u + b*n_s*n_c,
																																														g  + b*n_s*n_c,
																																														g_data + b*n_s*n_c,
																																														rx_cost  + b*n_s*n_r,
																																														ry_cost  + b*n_s*n_r,
																																														rz_cost  + b*n_s*n_r,
																																														g_rx + b*n_s*n_r,
																																														g_ry + b*n_s*n_c,
																																														g_rz + b*n_s*n_c));
    for(int b = 0; b < n_batches; b++)
        threads[b]->join();
    for(int b = 0; b < n_batches; b++)
        delete threads[b];
    delete threads;
      
    TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
      
    //clear unusable derviative
    for(int i = 0; i < sizes[6]; i++)
        g_par[i] = g_didx[i] = 0;
      
  }
  int num_buffers_full(){ return 0; }
  int num_buffers_branch(){ return 0; }
  int num_buffers_data(){ return 0; }
  int num_buffers_images(){ return 0; }
};
