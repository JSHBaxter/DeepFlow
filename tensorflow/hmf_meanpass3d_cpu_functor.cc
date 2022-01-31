
#include <thread>
#include "hmf_trees.h"
#include "hmf_meanpass3d_cpu_solver.h"


template <>
struct HmfMeanpass3dFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[7],
      const int* parentage,
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
    TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, sizes[6], sizes[4]);
    //node->print_tree();
    //TreeNode::print_list(bottom_up_list, sizes[6]+1);
    //std::cout << "Tree built" << std::endl;

    int n_batches = sizes[0];
	int n_s = sizes[1]*sizes[2]*sizes[3];
	int n_c = sizes[4];
	int n_r = sizes[6];
    int data_sizes[3] = {sizes[1],sizes[2],sizes[3]};
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(HMF_MEANPASS_CPU_SOLVER_3D(false,bottom_up_list, b, n_c, n_r, data_sizes,
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
      float** /*unused full buffers*/,
      float** /*unused image buffers*/){
      

    //build the tree
    TreeNode* node = NULL;
    TreeNode** children = NULL;
    TreeNode** bottom_up_list = NULL;
    TreeNode** top_down_list = NULL;
    TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, sizes[6], sizes[4]);
    //node->print_tree();
    //print_list(bottom_up_list, sizes[6]+1);
    //std::cout << "Tree built" << std::endl;
      
    int n_batches = sizes[0];
	int n_s = sizes[1]*sizes[2]*sizes[3];
	int n_c = sizes[4];
	int n_r = sizes[6];
    int data_sizes[3] = {sizes[1],sizes[2],sizes[3]};
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(HMF_MEANPASS_CPU_GRADIENT_3D(false,bottom_up_list, b, n_c, n_r, data_sizes,
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
        g_par[i] = 0;
      
  }
  int num_buffers_full(){ return 0; }
  int num_buffers_branch(){ return 0; }
  int num_buffers_data(){ return 0; }
  int num_buffers_images(){ return 0; }
};
