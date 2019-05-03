#include <math.h>
#include <thread>
#include <iostream>
#include <limits>

#include "cpu_kernels.h"

class PottsAuglag2dFunctor_SolverBatchThreadChannelsLast
{
private:
    const int b;
    const int n_x;
    const int n_y;
    const int n_c;
    const int n_s;
    const float* data;
    const float* rx;
    const float* ry;
    float* u;
    
public:
    PottsAuglag2dFunctor_SolverBatchThreadChannelsLast(
        const int batch,
        const int sizes[4],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        float* u ) :
    b(batch),
    n_x(sizes[1]),
    n_y(sizes[2]),
    n_c(sizes[3]),
    n_s(sizes[1]*sizes[2]),
    data(data_cost),
    rx(rx_cost),
    ry(ry_cost),
    u(u)
    {}
    
    void operator()(){
    
        // optimization constants
        const float beta = 0.001f;
        const float tau = 0.1f;
        const float cc = 0.125;
        const float icc = 1.0f / cc;
        
        //buffers shifted to correct batch
        const float* data_b = data + b*n_s*n_c;
        const float* rx_b = rx + b*n_s*n_c;
        const float* ry_b = ry + b*n_s*n_c;
        float* u_b = u + b*n_s*n_c;
        
        // allocate intermediate variables
        float max_change = 0.0f;
        float* ps = new float[n_s];
        float* pt = new float[n_s*n_c];
        float* g = new float[n_s*n_c];
        float* div = new float[n_s*n_c];
        float* px = new float[n_s*n_c];
        float* py = new float[n_s*n_c];
        float* pz = new float[n_s*n_c];

        //initialize variables
        clear(u_b, n_c*n_s);
        //softmax(data_b, u_b, n_s, n_c);
        clear(g, div, n_c*n_s);
        clear(px, py, pz, n_c*n_s);
        //clear(ps, n_s);
        //clear(pt, n_s*n_c);
        init_flows(data_b,ps,pt,n_s,n_c);

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
                compute_capacity_potts(g, u_b, ps, pt, div, n_s, n_c, tau, icc);
                compute_flows(g, div, px, py, rx_b, ry_b, n_c, n_x, n_y);
                compute_source_sink_multipliers( g, u_b, ps, pt, div, data_b, cc, icc, n_c, n_s);
                max_change = maxabs(g, n_c*n_s);
                std::cout << "Thread #:" << b << "\tIter #: " << iter << " \tMax change: " << max_change << std::endl;
            }

            max_change = maxabs(g, n_c*n_s);
            if (max_change < tau*beta)
                break;
        }

        //run one last block, just to be safe
        for (int iter = 0; iter < min_iter; iter++){
            compute_capacity_potts(g, u_b, ps, pt, div, n_s, n_c, tau, icc);
            compute_flows(g, div, px, py, rx_b, ry_b, n_c, n_x, n_y);
            compute_source_sink_multipliers( g, u_b, ps, pt, div, data_b, cc, icc, n_c, n_s);
        }
        
        //get final output
        log_buffer(u_b,n_s*n_c);
        
        //deallocate temporary buffers
        free(ps);
        free(g);
        free(pt);
        free(div);
        free(px);
        free(py);
    
    }
};



template <>
struct PottsAuglag2dFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[4],
      const float* data_cost,
      const float* rx_cost,
      const float* ry_cost,
      float* u,
      float** /*unused full buffers*/,
      float** /*unused image buffers*/){
      

    int n_batches = sizes[0];
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(PottsAuglag2dFunctor_SolverBatchThreadChannelsLast(b, sizes, data_cost, rx_cost, ry_cost, u));
    for(int b = 0; b < n_batches; b++)
        threads[b]->join();
    for(int b = 0; b < n_batches; b++)
        delete threads[b];
    delete threads;
      
  }
  int num_buffers_full(){ return 0; }
  int num_buffers_images(){ return 0; }
};