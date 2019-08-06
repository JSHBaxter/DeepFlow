#include <math.h>
#include <thread>
#include <iostream>
#include <limits>
#include "hmf_auglag_cpu_solver.h"
#include "cpu_kernels.h"
#include "hmf_trees.h"


class HMF_AUGLAG_CPU_SOLVER_2D : public HMF_AUGLAG_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const float* const rx;
    const float* const ry;
    float* px;
    float* py;
    float* rx_b;
    float* ry_b;

protected:

    int min_iter_calc(){
        return n_x + n_y;
    }
    
    virtual void clear_spatial_flows(){
        if( !px ) px = new float [n_s*n_r];
        if( !py ) py = new float [n_s*n_r];
        clear(px, py, n_r*n_s);
        
        if( !rx_b ) rx_b = new float [n_s*n_r];
        if( !ry_b ) ry_b = new float [n_s*n_r];
        for(int s = 0; s < n_s; s++){
            for(int r = 0; r < n_r; r++)
                rx_b[r*n_s+s] = rx[s*n_r+r];
            for(int r = 0; r < n_r; r++)
                ry_b[r*n_s+s] = ry[s*n_r+r];
        }
            
    }
    virtual void update_spatial_flow_calc(){
        compute_flows_channels_first(g, div, px, py, rx_b, ry_b, n_r, n_x, n_y);
    }
    
public:
    HMF_AUGLAG_CPU_SOLVER_2D(
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[6],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        float* u ) :
    HMF_AUGLAG_CPU_SOLVER_BASE(bottom_up_list,
                               batch,
                               sizes[1]*sizes[2],
                               sizes[3],
                               sizes[5],
                               data_cost,
                               u),
    n_x(sizes[1]),
    n_y(sizes[2]),
    rx(rx_cost),
    ry(ry_cost),
    px(0),
    py(0),
    rx_b(0),
    ry_b(0)
    {}
    
    void clean_up(){
        if( px ) delete px; px = 0;
        if( py ) delete py; py = 0;
        if( rx_b ) delete rx_b; rx_b = 0;
        if( ry_b ) delete ry_b; ry_b = 0;
    }
};

template <>
struct HmfAuglag2dFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[6],
      const int* parentage,
      const int* data_index,
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
    TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, data_index, sizes[5], sizes[3]);
    //node->print_tree();
    //TreeNode::print_list(bottom_up_list, sizes[7]+1);
    //std::cout << "Tree built" << std::endl;
      
    int n_batches = sizes[0];
    int n_s = sizes[1]*sizes[2];
    int n_c = sizes[3];
    int n_r = sizes[5];
    std::thread** threads = new std::thread* [n_batches];
    std::cout << threads << std::endl;
    HMF_AUGLAG_CPU_SOLVER_2D** solvers = new HMF_AUGLAG_CPU_SOLVER_2D* [n_batches];
    std::cout << solvers << std::endl;
    for(int b = 0; b < n_batches; b++){
        solvers[b] = new HMF_AUGLAG_CPU_SOLVER_2D(bottom_up_list, b, sizes, 
                                                  data_cost+b*n_s*n_c,
                                                  rx_cost+b*n_s*n_r,
												  ry_cost+b*n_s*n_r,
												  u+b*n_s*n_c);
        std::cout << solvers[b] << std::endl;
        threads[b] = new std::thread(*(solvers[b]));
        std::cout << threads[b] << std::endl;
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
