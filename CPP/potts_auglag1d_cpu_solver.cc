#include <math.h>
#include <thread>
#include <iostream>
#include <limits>

#include "cpu_kernels.h"

class PottsAuglag1dFunctor_SolverBatchThreadChannelsLast
{
private:
    const int b;
    const int n_x;
    const int n_c;
    const int n_s;
    const float* data;
    const float* rx;
    float* u;
    
public:
    PottsAuglag1dFunctor_SolverBatchThreadChannelsLast(
        const int batch,
        const int sizes[5],
        const float* data_cost,
        const float* rx_cost,
        float* u ) :
    b(batch),
    n_x(sizes[1]),
    n_c(sizes[2]),
    n_s(sizes[1]),
    data(data_cost),
    rx(rx_cost),
    u(u)
    {}
    
    void operator()(){
    
        // optimization constants
        const float beta = 0.01f;
        const float tau = 0.1f;
        const float cc = 0.125;
        const float icc = 1.0f / cc;
        
        //buffers shifted to correct batch
        const float* data_b = data + b*n_s*n_c;
        const float* rx_b = rx + b*n_s*n_c;
        float* u_b = u + b*n_s*n_c;
        
        // allocate intermediate variables
        float max_change = 0.0f;
        float* ps = new float[n_s];
        float* pt = new float[n_s*n_c];
        float* g = new float[n_s*n_c];
        float* div = new float[n_s*n_c];
        float* px = new float[n_s*n_c];

        //initialize variables
        clear(u_b, n_c*n_s);
        //softmax(data_b, u_b, n_s, n_c);
        clear(g, div, n_c*n_s);
        clear(px, n_c*n_s);
        clear(ps, n_s);
        clear(pt, n_s*n_c);
        //init_flows(data_b,ps,pt,n_s,n_c);

        // iterate in blocks
        int min_iter = 10;
        if (n_x > min_iter)
            min_iter = n_x;
        int max_loop = 200;
        for(int i = 0; i < max_loop; i++){    

            //run the solver a set block of iterations
            for (int iter = 0; iter < min_iter; iter++){
                compute_capacity_potts(g, u_b, ps, pt, div, n_s, n_c, tau, icc);
                compute_flows( g, div, px, rx_b, n_c, n_x);
                compute_source_sink_multipliers( g, u_b, ps, pt, div, data_b, cc, icc, n_c, n_s);
                max_change = maxabs(g, n_c*n_s);
                std::cout << "Iter #: " << iter << " Max change: " << max_change << std::endl;
            }

            max_change = maxabs(g, n_c*n_s);
            if (max_change < tau*beta)
                break;
        }

        //run one last block, just to be safe
        for (int iter = 0; iter < min_iter; iter++){
            compute_capacity_potts(g, u_b, ps, pt, div, n_s, n_c, tau, icc);
            compute_flows( g, div, px, rx_b, n_c, n_x);
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
    
    }
};



template <>
struct PottsAuglag1dFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[5],
      const float* data_cost,
      const float* rx_cost,
      float* u,
      float** /*unused full buffers*/,
      float** /*unused image buffers*/){
      

    int n_batches = sizes[0];
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(PottsAuglag1dFunctor_SolverBatchThreadChannelsLast(b, sizes, data_cost, rx_cost, u));
    for(int b = 0; b < n_batches; b++)
        threads[b]->join();
    for(int b = 0; b < n_batches; b++)
        delete threads[b];
    delete threads;
      
  }
  int num_buffers_full(){ return 0; }
  int num_buffers_images(){ return 0; }
};