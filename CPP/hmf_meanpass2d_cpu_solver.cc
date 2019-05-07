#include <math.h>
#include <thread>
#include <iostream>
#include <limits>
#include "cpu_kernels.h"
#include "hmf_trees.h"


namespace HMF2DMP_CPU {
    
class SolverBatchThreadChannelsLast
{
private:
    TreeNode const* const* bottom_up_list;
    const int b;
    const int n_x;
    const int n_y;
    const int n_c;
    const int n_r;
    const int n_s;
    const float* data;
    const float* rx;
    const float* ry;
    float* u;

    inline int idx (const int x, const int y){
        return y + this->n_y * x;
    }
    inline int idxc (const int s, const int c){
        return c + this->n_c*s;
    }
    inline int idxc (const int x, const int y, const int c){
        return c + this->n_c*(idx(x,y));
    }
    inline int idxc (const int b, const int x, const int y, const int c){
        return (b*this->n_s*this->n_c) + idxc(x,y,c);
    }
    inline int idxr (const int s, const int c){
        return c + this->n_r*s;
    }
    inline int idxr (const int x, const int y, const int c){
        return c + this->n_r*(idx(x,y));
    }
    inline int idxr (const int b, const int x, const int y, const int c){
        return (b*this->n_s*this->n_r) + idxr(x,y,c);
    }
    
public:
    SolverBatchThreadChannelsLast(
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[6],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        float* u ) :
    bottom_up_list(bottom_up_list),
    b(batch),
    n_x(sizes[1]),
    n_y(sizes[2]),
    n_c(sizes[3]),
    n_r(sizes[5]),
    n_s(sizes[1]*sizes[2]),
    data(data_cost),
    rx(rx_cost),
    ry(ry_cost),
    u(u)
    {}
    
    
    //perform tree-wise aggregation within buffer
    void aggregate_top_down(float* bufferout){
        for (int s = 0; s < n_s; s++)
            for (int l = n_r-1; l >= 0; l--) {
                const TreeNode* n = bottom_up_list[l];
                for(int c = 0; c < n->c; c++)
                    bufferout[idxr(s,n->children[c]->r)] += bufferout[idxr(s,n->r)];
            }
    }
    
    
    void operator()(){
        
        const float* data_b = data + b*n_s*n_c;
        const float* rx_b = rx + b*n_s*n_r;
        const float* ry_b = ry + b*n_s*n_r;
        float* u_b = u + b*n_s*n_c;
        
        // optimization constants
        const float tau = 0.5f;
        const float beta = 0.02f;
        const float epsilon = 10e-5f;
        
        // allocate intermediate variables
        float max_change = 0.0f;
        float* u_tmp = new float[n_s*n_r];
        float* r_eff = new float[n_s*n_r];

        //initialize variables
        softmax(data_b, u_b, n_s, n_c);
        
        // iterate in blocks
        int min_iter = 10;
        if (n_x > min_iter)
            min_iter = n_x;
        if (n_y > min_iter)
            min_iter = n_y;
        int max_loop = 200;
        for(int i = 0; i < max_loop; i++){
            
            //run the solver a set block of iterations
            for (int iter = 0; iter < min_iter; iter++){
                aggregate_bottom_up(u_b,u_tmp,n_s,n_c,n_r,bottom_up_list);
                calculate_r_eff(r_eff, rx_b, ry_b, u_tmp, n_x, n_y, n_r);
                aggregate_top_down(r_eff);
                for(int s = 0, i = 0; s < n_s; s++)
                    for(int c = 0; c < n_c; c++, i++)
                    r_eff[i] = data_b[i]+r_eff[idxr(s,c)];
                max_change = softmax_with_convergence(r_eff, u_b, n_s, n_c, tau);
            }

            if (max_change < beta)
                break;
        }

        //run one last block, just to be safe
        for (int iter = 0; iter < min_iter; iter++){
            aggregate_bottom_up(u_b,u_tmp,n_s,n_c,n_r,bottom_up_list);
            calculate_r_eff(r_eff, rx_b, ry_b, u_tmp, n_x, n_y, n_r);
            aggregate_top_down(r_eff);
            for(int s = 0, i = 0; s < n_s; s++)
                for(int c = 0; c < n_c; c++, i++)
                r_eff[i] = data_b[i]+r_eff[idxr(s,c)];
            max_change = softmax_with_convergence(r_eff, u_b, n_s, n_c, tau);
        }
        
        //perform majority of last iteration
        aggregate_bottom_up(u_b,u_tmp,n_s,n_c,n_r,bottom_up_list);
        calculate_r_eff(r_eff, rx_b, ry_b, u_tmp, n_x, n_y, n_r);
        aggregate_top_down(r_eff);

        //get final output
        for(int s = 0; s < n_s; s++)
            for(int c = 0; c < n_c; c++)
                u_b[idxc(s,c)] = data_b[idxc(s,c)]+r_eff[idxr(s,c)];
        
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
    const int n_c;
    const int n_r;
    const int n_s;
    float* g_data;
    float* g_rx;
    float* g_ry;
    const float* u;
    const float* grad;
    
public:
    GradientBatchThreadChannelsLast(
        TreeNode** bottom_up_list,
        const int batch,
        const int sizes[6],
        const float* u,
        const float* g,
        float* g_d,
        float* g_rx,
        float* g_ry ) :
    bottom_up_list(bottom_up_list),
    b(batch),
    n_x(sizes[1]),
    n_y(sizes[2]),
    n_c(sizes[3]),
    n_r(sizes[5]),
    n_s(sizes[1]*sizes[2]),
    g_data(g_d),
    g_rx(g_rx),
    g_ry(g_ry),
    u(u),
    grad(g)
    {}


    inline int idx (const int x, const int y){
        return y + this->n_y * x;
    }
    inline int idxc (const int s, const int c){
        return c + this->n_c*s;
    }
    inline int idxc (const int x, const int y, const int c){
        return c + this->n_c*(idx(x,y));
    }
    inline int idxc (const int b, const int x, const int y, const int c){
        return (b*this->n_s*this->n_c) + idxc(x,y,c);
    }
    inline int idxr (const int s, const int c){
        return c + this->n_r*s;
    }
    inline int idxr (const int x, const int y, const int c){
        return c + this->n_r*(idx(x,y));
    }
    inline int idxr (const int b, const int x, const int y, const int c){
        return (b*this->n_s*this->n_r) + idxr(x,y,c);
    }
    
    void operator()(){
        
        float* g_d_b = g_data + b*n_s*n_c;
        float* g_rx_b = g_rx + b*n_s*n_c;
        float* g_ry_b = g_ry + b*n_s*n_c;
        const float* g_b = grad + b*n_s*n_c;
        const float* u_b = u + b*n_s*n_c;
        float* g_r_eff = g_d_b; 

        float epsilon = 10e-5f;
        
        // populate data gradient
        for (int s = 0; s < n_s; s++)
            for (int c = 0; c < n_c; c++)
                g_d_b[idxc(s,c)] = g_b[idxc(s,c)];
        
        for (int x = 0; x < n_x; x++)
        for (int y = 0; y < n_y; y++) {

            // populate rx gradient from rf gradient
            for (int c = 0; c < n_c; c++){
                g_rx_b[idxr(x,y,c)] = 0.0f;
                if (x < n_x-1){
                    g_rx_b[idxr(x,y,c)] += g_r_eff[idxr(x,y,c)] * u_b[idxc(x+1,y,c)];
                    g_rx_b[idxr(x,y,c)] += g_r_eff[idxr(x+1,y,c)] * u_b[idxc(x,y,c)];
                }
                g_rx_b[idxr(x,y,c)] *= 0.5f;
            }

            // populate ry gradient
            for (int c = 0; c < n_c; c++){
                g_ry_b[idxr(x,y,c)] = 0.0f;
                if (y < n_y-1){
                    g_ry_b[idxr(x,y,c)] += g_r_eff[idxr(x,y,c)] * u_b[idxc(x,y+1,c)];
                    g_ry_b[idxr(x,y,c)] += g_r_eff[idxr(x,y+1,c)] * u_b[idxc(x,y,c)];
                }
                g_ry_b[idxr(x,y,c)] *= 0.5f;
            }

        }
        
    }
};


}

template <>
struct HmfMeanpass2dFunctor<CPUDevice> {
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
    //TreeNode::print_list(bottom_up_list, sizes[6]+1);
    //std::cout << "Tree built" << std::endl;

    int n_batches = sizes[0];
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(HMF2DMP_CPU::SolverBatchThreadChannelsLast(bottom_up_list, b, sizes, data_cost, rx_cost, ry_cost, u));
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
struct HmfMeanpass2dGradFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[6],
      const int* parentage,
      const int* data_index,
      const float* data_cost,
      const float* rx_cost,
      const float* ry_cost,
      const float* u,
      const float* g,
      float* g_data,
      float* g_rx,
      float* g_ry,
      int* g_par,
      int* g_didx,
      float** /*unused full buffers*/,
      float** /*unused image buffers*/){
      

    //build the tree
    TreeNode* node = NULL;
    TreeNode** children = NULL;
    TreeNode** bottom_up_list = NULL;
    TreeNode** top_down_list = NULL;
    TreeNode::build_tree(node, children, bottom_up_list, top_down_list, parentage, data_index, sizes[5], sizes[3]);
    //node->print_tree();
    //print_list(bottom_up_list, sizes[6]+1);
    //std::cout << "Tree built" << std::endl;
      
    int n_batches = sizes[0];
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(HMF2DMP_CPU::GradientBatchThreadChannelsLast(bottom_up_list, b, sizes, u, g, g_data, g_rx, g_ry));
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
