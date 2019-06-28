#include <math.h>
#include <thread>
#include <iostream>
#include <limits>
#include "hmf_auglag_cpu_solver.h"
#include "cpu_kernels.h"
#include "hmf_trees.h"


class HMF_AUGLAG_CPU_SOLVER_1D : public HMF_AUGLAG_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const float* const rx;
    float* const px;
    float* const rx_b;

protected:

    int min_iter_calc(){
        return n_x;
    }
    
    virtual void clear_spatial_flows(){
        clear(px, n_r*n_s);
        
        for(int s = 0; s < n_s; s++)
            for(int r = 0; r < n_r; r++)
                rx_b[r*n_s+s] = rx[s*n_r+r];
            
    }
    virtual void update_spatial_flow_calc(){
        compute_flows_channels_first(g, div, px, rx_b, n_r, n_x);
    }
    
public:
    HMF_AUGLAG_CPU_SOLVER_1D(
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[5],
        const float* data_cost,
        const float* rx_cost,
        float* u ) :
    HMF_AUGLAG_CPU_SOLVER_BASE(bottom_up_list,
                               batch,
                               sizes[1],
                               sizes[2],
                               sizes[4],
                               data_cost,
                               u),
    n_x(sizes[1]),
    rx(rx_cost+batch*n_s*n_r),
    px(new float[n_s*n_r]),
    rx_b(new float[n_s*n_r])
    {
        std::cout << "Derived class:" << std::endl;
        std::cout << "\t" << px << std::endl;
        std::cout << "\t" << rx_b << std::endl;
    }
    
    ~HMF_AUGLAG_CPU_SOLVER_1D(){
        //free(px);
        //free(rx_b);
    }
};

template <>
struct HmfAuglag1dFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[5],
      const int* parentage,
      const int* data_index,
      const float* data_cost,
      const float* rx_cost,
      float* u,
      float** /*unused full buffers*/,
      float** /*unused image buffers*/){
      
    //build the tree
    TreeNode* node = NULL;
    TreeNode** children = NULL;
    TreeNode** bottom_up_list = NULL;
    TreeNode** top_down_list = NULL;
    TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, data_index, sizes[4], sizes[2]);
    //node->print_tree();
    //TreeNode::print_list(bottom_up_list, sizes[7]+1);
    //std::cout << "Tree built" << std::endl;
      
    int n_batches = sizes[0];
    std::thread** threads = new std::thread* [n_batches];
    std::cout << threads << std::endl;
    HMF_AUGLAG_CPU_SOLVER_1D** solvers = new HMF_AUGLAG_CPU_SOLVER_1D* [n_batches];
    std::cout << solvers << std::endl;
    for(int b = 0; b < n_batches; b++){
        solvers[b] = new HMF_AUGLAG_CPU_SOLVER_1D(bottom_up_list, b, sizes, data_cost, rx_cost, u);
        std::cout << solvers[b] << std::endl;
        threads[b] = new std::thread(*(solvers[b]));
        std::cout << threads[b] << std::endl;
    }
    for(int b = 0; b < n_batches; b++)
        //(*(solvers[b]))();
        threads[b]->join();
    for(int b = 0; b < n_batches; b++){
        //delete threads[b];
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
