#include <math.h>
#include <thread>
#include <iostream>
#include <limits>
#include "cpu_kernels.h"
#include "hmf_trees.h"


namespace HMF2DAL_CPU {
    
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
    const float* const data;
    const float* const rx;
    const float* const ry;
    float* const ps;
    float* const pt;
    float* const px;
    float* const py;
    float* const u_tmp;
    float* const div;
    float* const g;
    float* const u;
    float* const data_b;
    float* const rx_b;
    float* const ry_b;
    
    // optimization constants
    const float tau = 0.1f;
    const float beta = 0.005f;
    const float epsilon = 10e-5f;
    const float cc;
    const float icc;
    
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
    n_s(n_x*n_y),
    data(data_cost),
    rx(rx_cost),
    ry(ry_cost),
    ps(new float[n_s]),
    u_tmp(new float[n_s*n_r]),
    pt(new float[n_s*n_r]),
    px(new float[n_s*n_r]),
    py(new float[n_s*n_r]),
    div(new float[n_s*n_r]),
    g(new float[n_s*n_r]),
    rx_b(new float[n_s*n_r]),
    ry_b(new float[n_s*n_r]),
    data_b(new float[n_s*n_r]),
    cc(1.0f / max_diff(data_cost,n_c,n_s)),
    icc(1.0f/cc),
    u(u)
    { }
    
    //perform one iteration of the algorithm
    void block_iter(){
                
        /*for(int s = 0; s < n_s; s++)
            std::cout << ps[s] <<  " ";
        std::cout << std::endl;
        for(int c = 0; c < n_r; c++)
        for(int s = 0; s < n_s; s++)
            std::cout << pt[c*n_s+s] <<  " ";
        std::cout << std::endl;
        for(int c = 0; c < n_r; c++)
        for(int s = 0; s < n_s; s++)
            std::cout << u_tmp[c*n_s+s] <<  " ";
        std::cout << std::endl;*/

        //calculate the capacity and then update flows
        //std::cout << "\tUpdate capacities" << std::endl;
        for(int n_n = 0; n_n < n_r; n_n++){
            const TreeNode* n = bottom_up_list[n_n];
            int r = n->r;
            if( n->parent->parent == NULL )
                compute_capacity_potts(g+r*n_s, u_tmp+r*n_s, ps, pt+r*n_s, div+r*n_s, n_s, 1, tau, icc);
            else
                compute_capacity_potts(g+r*n_s, u_tmp+r*n_s, pt+n->parent->r*n_s, pt+r*n_s, div+r*n_s, n_s, 1, tau, icc);
        }
        //std::cout << "\tUpdate flow" << std::endl;
        compute_flows_channels_first(g, div, px, py, rx_b, ry_b, n_r, n_x, n_y);
        
        //update source and sink multipliers top down
        //std::cout << "\tUpdate source/sink flows" << std::endl;
        for(int n_n = 0; n_n < n_r+1; n_n++){
            const TreeNode* n = bottom_up_list[n_r-n_n];
            float* n_pt_buf = pt+n->r*n_s;
            float* n_g_buf = g+n->r*n_s;
            float* n_div_buf = div+n->r*n_s;
            float* n_u_buf = u_tmp+n->r*n_s;
            float* n_d_buf = data_b+n->r*n_s;

            //if we are the source node
            if(n->r == -1){
                //std::cout << "\t\tSource" << std::endl;
                set(ps, icc, n_s);
                for(int c = 0; c < n->c; c++){
                    const TreeNode* nc = n->children[c];
                    float* c_pt_buf = pt+nc->r*n_s;
                    float* c_div_buf = div+nc->r*n_s;
                    float* c_u_buf = u_tmp+nc->r*n_s;
                    inc(c_pt_buf, ps, n_s);
                    inc(c_div_buf, ps, n_s);
                    inc(c_u_buf, ps, -icc, n_s);
                }
                mult_buffer(ps, 1.0f / (float) n->c, n_s);
            }

            //if we are a branch node
            else if(n->c > 0){
                //std::cout << "\t\tBranch" << std::endl;
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
                }
                mult_buffer(n_pt_buf, 1.0f / (float) (n->c+1), n_s);

            }

            //if we are a leaf node
            else{
                //std::cout << "\t\tLeaf" << std::endl;
                const TreeNode* p = n->parent;
                float* p_pt_buf = pt+p->r*n_s;
                if( p->r == -1 )
                    p_pt_buf = ps;
                copy(p_pt_buf,n_pt_buf,n_s);
                ninc(n_div_buf, n_pt_buf, n_s);
                inc(n_u_buf, n_pt_buf, icc, n_s);
                constrain(n_pt_buf,n_d_buf,n_s);

            }
        }

        //update multipliers
        //std::cout << "\tUpdate multipliers" << std::endl;
        for(int n_n = 0; n_n < n_r; n_n++){
            const TreeNode* n = bottom_up_list[n_n];
            const TreeNode* p = n->parent;
            float* n_pt_buf = pt+n->r*n_s;
            float* n_g_buf = g+n->r*n_s;
            float* n_div_buf = div+n->r*n_s;
            float* n_u_buf = u_tmp+n->r*n_s;
            float* p_pt_buf = pt+p->r*n_s;
            if( p->r == -1 )
                p_pt_buf = ps;
            copy(n_pt_buf,n_g_buf,n_s);
            inc(n_div_buf,n_g_buf,n_s);
            ninc(p_pt_buf,n_g_buf,n_s);
            mult_buffer(n_g_buf, cc, n_s);
            ninc(n_g_buf,n_u_buf,n_s);
        }
    }
    
    void operator()(){
        
        std::cout << cc << std::endl;
        
        // transpose input data (makes everything easier)
        for(int s = 0; s < n_s; s++){
            for(int c = 0; c < n_c; c++)
                data_b[c*n_s+s] = -(data + b*n_s*n_c)[s*n_c+c];
            for(int r = 0; r < n_r; r++)
                rx_b[r*n_s+s] = (rx + b*n_s*n_r)[s*n_r+r];
            for(int r = 0; r < n_r; r++)
                ry_b[r*n_s+s] = (ry + b*n_s*n_r)[s*n_r+r];
        }
        
        //initialize variables
        std::cout << "Init variables" << std::endl;
        std::cout << n_r << std::endl;
        clear(g, div, u_tmp, n_r*n_s);
        clear(px, py, pt, n_r*n_s);
        clear(ps, n_s);
        init_flows_channels_first(data_b, ps, n_c, n_s);
        for(int i = 0; i < n_r; i++)
            copy(ps,pt+i*n_s,n_s);
        
        // iterate in blocks
        int min_iter = 10;
        if (n_x+n_y > min_iter)
            min_iter = n_x+n_y;
        int max_loop = 200;
        for(int i = 0; i < max_loop; i++){
            
            //run the solver a set block of iterations
            for (int iter = 0; iter < min_iter; iter++){
                
                std::cout << "Iter " << i << " - " << iter << std::endl;
                block_iter();
                
                
            }

            float max_change = maxabs(g,n_s*n_r);
            std::cout << "Calculate max change: " << max_change << std::endl;
            if (max_change < beta)
                break;
        }

        //run one last block, just to be safe
        for (int iter = 0; iter < min_iter; iter++){
            std::cout << "Iter " << "LAST - " << iter << std::endl;
            block_iter();
        }

        //log output and transpose output back into proper buffer
        //log_buffer(u_tmp, n_s*n_c);
        float* u_b = u + b*n_s*n_c;
        for(int s = 0; s < n_s; s++)
            for(int c = 0; c < n_c; c++)
                u_b[s*n_c+c] = u_tmp[c*n_s+s];
        
        //deallocate temporary buffers
        free(u_tmp);
        free(rx_b);
        free(ry_b);
        free(pt);
        free(px);
        free(py);
        free(ps);
        free(g);
        free(div);
        free(data_b);
    }
};

}

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
    //TreeNode::print_list(bottom_up_list, sizes[6]+1);
    //std::cout << "Tree built" << std::endl;

    int n_batches = sizes[0];
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(HMF2DAL_CPU::SolverBatchThreadChannelsLast(bottom_up_list, b, sizes, data_cost, rx_cost, ry_cost, u));
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
