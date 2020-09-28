#include <math.h>
#include <thread>
#include <iostream>
#include <limits>
#include "hmf_auglag_cpu_solver.h"
#include "cpu_kernels.h"
#include "hmf_trees.h"
#include <algorithm>


class HMF_AUGLAG_CPU_SOLVER_3D : public HMF_AUGLAG_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const int n_z;
    const float* const rx;
    const float* const ry;
    const float* const rz;
    float* px;
    float* py;
    float* pz;
    float* rx_b;
    float* ry_b;
    float* rz_b;

protected:

    int min_iter_calc(){
        return std::max(n_x,std::max(n_y,n_z))+n_r-n_c;
    }
    
    virtual void clear_spatial_flows(){
        if( !px ) px = new float [n_s*n_r];
        if( !py ) py = new float [n_s*n_r];
        if( !pz ) pz = new float [n_s*n_r];
        clear(px, py, pz, n_r*n_s);
        
        if( !rx_b ) rx_b = new float [n_s*n_r];
        if( !ry_b ) ry_b = new float [n_s*n_r];
        if( !rz_b ) rz_b = new float [n_s*n_r];
        for(int s = 0; s < n_s; s++){
            for(int r = 0; r < n_r; r++)
                rx_b[r*n_s+s] = rx[s*n_r+r];
            for(int r = 0; r < n_r; r++)
                ry_b[r*n_s+s] = ry[s*n_r+r];
            for(int r = 0; r < n_r; r++)
                rz_b[r*n_s+s] = rz[s*n_r+r];
        }
            
    }
    virtual void update_spatial_flow_calc(){
        compute_flows_channels_first(g, div, px, py, pz, rx_b, ry_b, rz_b, n_r, n_x, n_y, n_z);
    }
    
public:
    HMF_AUGLAG_CPU_SOLVER_3D(
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[7],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
        float* u ) :
    HMF_AUGLAG_CPU_SOLVER_BASE(bottom_up_list,
                               batch,
                               sizes[1]*sizes[2]*sizes[3],
                               sizes[4],
                               sizes[6],
                               data_cost,
                               u),
    n_x(sizes[1]),
    n_y(sizes[2]),
    n_z(sizes[3]),
    rx(rx_cost),
    ry(ry_cost),
    rz(rz_cost),
    px(0),
    py(0),
    pz(0),
    rx_b(0),
    ry_b(0),
    rz_b(0)
    {
    }
    
    void clean_up(){
        if( px ) delete px; px = 0;
        if( py ) delete py; py = 0;
        if( pz ) delete pz; pz = 0;
        if( rx_b ) delete rx_b; rx_b = 0;
        if( ry_b ) delete ry_b; ry_b = 0;
        if( rz_b ) delete rz_b; rz_b = 0;
    }
};

template <>
struct HmfAuglag3dFunctor<CPUDevice> {
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
    //TreeNode::print_list(bottom_up_list, sizes[7]+1);
    //std::cout << "Tree built" << std::endl;
      
    int n_batches = sizes[0];
    int n_s = sizes[1]*sizes[2]*sizes[3];
    int n_c = sizes[4];
    int n_r = sizes[6];
    std::thread** threads = new std::thread* [n_batches];
    HMF_AUGLAG_CPU_SOLVER_3D** solvers = new HMF_AUGLAG_CPU_SOLVER_3D* [n_batches];
    std::cout << solvers << std::endl;
    for(int b = 0; b < n_batches; b++){
        solvers[b] = new HMF_AUGLAG_CPU_SOLVER_3D(bottom_up_list, b, sizes, 
                                                  data_cost+b*n_s*n_c,
                                                  rx_cost+b*n_s*n_r,
												  ry_cost+b*n_s*n_r,
												  rz_cost+b*n_s*n_r,
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
