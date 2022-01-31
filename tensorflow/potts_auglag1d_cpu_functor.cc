#include "potts_auglag1d_cpu_solver.h"
#include <thread>

template <>
struct PottsAuglag1dFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[3],
      const float* data_cost,
      const float* rx_cost,
      float* u,
      float** /*unused full buffers*/,
      float** /*unused image buffers*/){
      

    int n_c = sizes[2];
    int n_s = sizes[1];
    int n_batches = sizes[0];
    int data_sizes[1] = {n_s};
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(POTTS_AUGLAG_CPU_SOLVER_1D(false, b, n_c, data_sizes, 
																data_cost+b*n_s*n_c,
																rx_cost+b*n_s*n_c,
																u+b*n_s*n_c));
    for(int b = 0; b < n_batches; b++)
        threads[b]->join();
    for(int b = 0; b < n_batches; b++)
        delete threads[b];
    delete threads;
      
  }
  int num_buffers_full(){ return 0; }
  int num_buffers_images(){ return 0; }
};