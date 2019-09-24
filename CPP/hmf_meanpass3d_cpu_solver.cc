#include <math.h>
#include <thread>
#include <iostream>
#include <limits>
#include "cpu_kernels.h"
#include "hmf_trees.h"
#include "hmf_meanpass_cpu_solver.h"
#include <algorithm>

class HMF_MEANPASS_CPU_SOLVER_3D : public HMF_MEANPASS_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const int n_z;
    const float* const rx;
    const float* const ry;
    const float* const rz;

protected:
    int min_iter_calc(){
        return std::max(n_x,std::max(n_y,n_z))+n_r-n_c;
    }
    
    void update_spatial_flow_calc(){
        calculate_r_eff(r_eff, rx, ry, rz, u_tmp, n_x, n_y, n_z, n_r);
    }
    
public:
    HMF_MEANPASS_CPU_SOLVER_3D(
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[7],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
		const float* init_u,
        float* u ) :
    HMF_MEANPASS_CPU_SOLVER_BASE(bottom_up_list,batch,
                                 sizes[1]*sizes[2]*sizes[3],
                                 sizes[4],
                                 sizes[6],
                                 data_cost,
								 init_u,
                                 u),
    n_x(sizes[1]),
    n_y(sizes[2]),
    n_z(sizes[3]),
    rx(rx_cost),
    ry(ry_cost),
    rz(rz_cost)
    {}
};

class HMF_MEANPASS_CPU_GRADIENT_3D : public HMF_MEANPASS_CPU_GRADIENT_BASE
{
private:
    const int n_x;
    const int n_y;
    const int n_z;
    float* g_rx;
    float* g_ry;
    float* g_rz;
	const float* rx;
	const float* ry;
	const float* rz;

protected:
    int min_iter_calc(){
        return n_x + n_y + n_z;
    }
    
    void update_spatial_flow_calc(bool use_tau){
        get_reg_gradients(dy, u, g_rx, g_ry, g_rz, n_x, n_y, n_z, n_r, use_tau ? tau : 1.0f);
        get_gradient_for_u(dy, rx, ry, rz, g_u, n_x, n_y, n_z, n_r, use_tau ? tau : 1.0f);
    }
    
public:
    HMF_MEANPASS_CPU_GRADIENT_3D(
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
    HMF_MEANPASS_CPU_GRADIENT_BASE(bottom_up_list,batch,
                                 sizes[1]*sizes[2]*sizes[3],
                                 sizes[4],
                                 sizes[6],
                                 u,
                                 g,
                                 g_d),
    n_x(sizes[1]),
    n_y(sizes[2]),
    n_z(sizes[3]),
    rx(rx_cost),
    ry(ry_cost),
    rz(rz_cost),
    g_rx(g_rx),
    g_ry(g_ry),
    g_rz(g_rz)
    {}
};

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
	  const float* init_u,
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
        threads[b] = new std::thread(HMF_MEANPASS_CPU_SOLVER_3D(bottom_up_list, b, sizes,
                                                                data_cost + b*n_s*n_c,
                                                                rx_cost + b*n_s*n_r,
                                                                ry_cost + b*n_s*n_r,
                                                                rz_cost + b*n_s*n_r,
																init_u + (init_u ? b*n_s*n_c : 0),
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
        threads[b] = new std::thread(HMF_MEANPASS_CPU_GRADIENT_3D(bottom_up_list, b, sizes,
                                                                  u + b*n_s*n_c,
                                                                  g + b*n_s*n_c,
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
