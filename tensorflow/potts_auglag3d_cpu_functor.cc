#include <thread>

#include "potts_auglag3d_cpu_solver.h"

template <>
struct PottsAuglag3dFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[5],
      const float* data_cost,
      const float* rx_cost,
      const float* ry_cost,
      const float* rz_cost,
      float* u,
      float** /*unused full buffers*/,
      float** /*unused image buffers*/){
      

    int n_c = sizes[4];
    int n_s = sizes[1]*sizes[2]*sizes[3];
    int n_batches = sizes[0];
    int data_sizes[3] = {sizes[1],sizes[2],sizes[3]};
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(POTTS_AUGLAG_CPU_SOLVER_3D(false, b, n_c, data_sizes,
																data_cost+b*n_s*n_c,
																rx_cost+b*n_s*n_c,
																ry_cost+b*n_s*n_c,
																rz_cost+b*n_s*n_c,
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
