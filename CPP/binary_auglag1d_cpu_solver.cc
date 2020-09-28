#include <math.h>
#include <thread>
#include <iostream>
#include <limits>

#include "binary_auglag_cpu_solver.h"
#include "cpu_kernels.h"

class BINARY_AUGLAG_CPU_SOLVER_1D : public BINARY_AUGLAG_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const float* rx;
	float* px;

protected:
    virtual int min_iter_calc(){
		return n_x;
	}
	
    virtual void clear_spatial_flows(){
        px = new float[n_s*n_c];
		clear(px, n_s*n_c);
	}
	
    virtual void update_spatial_flow_calc(){
		compute_flows( g, div, px, rx, n_c, n_x);
	}
	
    virtual void clean_up(){
		if( px ) delete px; px = 0;
	}

public:
	BINARY_AUGLAG_CPU_SOLVER_1D(
        const int batch,
        const int sizes[3],
        const float* data_cost,
        const float* rx_cost,
        float* u 
	):
	BINARY_AUGLAG_CPU_SOLVER_BASE(batch, sizes[1], sizes[2], data_cost, u),
    n_x(sizes[1]),
    rx(rx_cost),
	px(0)
	{}
};

template <>
struct BinaryAuglag1dFunctor<CPUDevice> {
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
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(BINARY_AUGLAG_CPU_SOLVER_1D(b, sizes, 
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
