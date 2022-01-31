
#include <thread>
#include "hmf_auglag2d_cpu_solver.h"


template <>
struct HmfAuglag2dFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[6],
      const int* parentage,
      const float* data_cost,
      const float* rx_cost,
      const float* ry_cost,
      float* u,
      float** /*unused full buffers*/,
      float** /*unused image buffers*/){
      
    //build the tree
    TreeNode* node = NULL;
    TreeNode** children = NULL;
    TreeNode** bottom_up_list = NULL;
    TreeNode** top_down_list = NULL;
    TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, sizes[5], sizes[3]);
    //node->print_tree();
    //TreeNode::print_list(bottom_up_list, sizes[7]+1);
    //std::cout << "Tree built" << std::endl;
      
    int n_batches = sizes[0];
    int n_s = sizes[1]*sizes[2];
    int n_c = sizes[3];
    int n_r = sizes[5];
    int data_sizes[2] = {sizes[1],sizes[2]};
    std::thread** threads = new std::thread* [n_batches];
    HMF_AUGLAG_CPU_SOLVER_2D** solvers = new HMF_AUGLAG_CPU_SOLVER_2D* [n_batches];
    for(int b = 0; b < n_batches; b++){
        solvers[b] = new HMF_AUGLAG_CPU_SOLVER_2D(false,bottom_up_list, b, n_c, n_r, data_sizes, 
                                                  data_cost+b*n_s*n_c,
                                                  rx_cost+b*n_s*n_r,
												  ry_cost+b*n_s*n_r,
												  u+b*n_s*n_c);
        threads[b] = new std::thread(*(solvers[b]));
    }
    for(int b = 0; b < n_batches; b++)
        //(*(solvers[b]))();
        threads[b]->join();
    for(int b = 0; b < n_batches; b++){
        delete threads[b];
        delete solvers[b];
    }
    delete threads;
    delete solvers;
      
    TreeNode::free_tree(node, children, bottom_up_list, top_down_list);
      
  }
  int num_buffers_full(){ return 0; }
  int num_buffers_branch(){ return 0; }
  int num_buffers_data(){ return 0; }
  int num_buffers_images(){ return 0; }
};
