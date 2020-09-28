#include <math.h>
#include <thread>
#include <iostream>
#include <limits>

#include "binary_auglag_cpu_solver.h"
#include "cpu_kernels.h"

class BINARY_AUGLAG_CPU_SOLVER_2D : public BINARY_AUGLAG_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const float* rx;
    const float* ry;
	float* px;
	float* py;

protected:
    virtual int min_iter_calc(){
		return std::max(n_x,n_y);
	}
	
    virtual void clear_spatial_flows(){
        px = new float[n_s*n_c];
        py = new float[n_s*n_c];
		clear(px, py, n_s*n_c);
	}
	
    virtual void update_spatial_flow_calc(){
		compute_flows( g, div, px, py, rx, ry, n_c, n_x, n_y);
	}
	
    virtual void clean_up(){
		if( px ) delete px; px = 0;
		if( py ) delete py; py = 0;
	}

public:
	BINARY_AUGLAG_CPU_SOLVER_2D(
        const int batch,
        const int sizes[4],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        float* u 
	):
	BINARY_AUGLAG_CPU_SOLVER_BASE(batch, sizes[1]*sizes[2], sizes[3], data_cost, u),
    n_x(sizes[1]),
    n_y(sizes[2]),
    rx(rx_cost),
    ry(ry_cost),
	px(0),
	py(0)
	{}
};

template <>
struct BinaryAuglag2dFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
      int sizes[4],
      const float* data_cost,
      const float* rx_cost,
      const float* ry_cost,
      float* u,
      float** /*unused full buffers*/,
      float** /*unused image buffers*/){
      

    int n_c = sizes[3];
    int n_s = sizes[1]*sizes[2];
    int n_batches = sizes[0];
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(BINARY_AUGLAG_CPU_SOLVER_2D(b, sizes,
																data_cost+b*n_s*n_c,
																rx_cost+b*n_s*n_c,
																ry_cost+b*n_s*n_c,
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
