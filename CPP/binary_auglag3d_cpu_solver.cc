#include <math.h>
#include <thread>
#include <iostream>
#include <limits>

#include "binary_auglag_cpu_solver.h"
#include "cpu_kernels.h"

class BINARY_AUGLAG_CPU_SOLVER_3D : public BINARY_AUGLAG_CPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const int n_z;
    const float* rx;
    const float* ry;
    const float* rz;
	float* px;
	float* py;
	float* pz;

protected:
    virtual int min_iter_calc(){
		return n_x+n_y+n_z;
	}
	
    virtual void clear_spatial_flows(){
        px = new float[n_s*n_c];
        py = new float[n_s*n_c];
        pz = new float[n_s*n_c];
		clear(px, py, pz, n_s*n_c);
	}
	
    virtual void update_spatial_flow_calc(){
		compute_flows(g, div, px, py, pz, rx, ry, rz, n_c, n_x, n_y, n_z);
	}
	
    virtual void clean_up(){
		if( px ) delete px; px = 0;
		if( py ) delete py; py = 0;
		if( pz ) delete pz; pz = 0;
	}

public:
	BINARY_AUGLAG_CPU_SOLVER_3D(
        const int batch,
        const int sizes[5],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        const float* rz_cost,
        float* u 
	):
	BINARY_AUGLAG_CPU_SOLVER_BASE(batch, sizes[1]*sizes[2]*sizes[3], sizes[4], data_cost, u),
    n_x(sizes[1]),
    n_y(sizes[2]),
    n_z(sizes[3]),
    rx(rx_cost),
    ry(ry_cost),
    rz(rz_cost),
	px(0),
	py(0),
	pz(0)
	{}
};



template <>
struct BinaryAuglag3dFunctor<CPUDevice> {
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
    std::thread** threads = new std::thread* [n_batches];
    for(int b = 0; b < n_batches; b++)
        threads[b] = new std::thread(BINARY_AUGLAG_CPU_SOLVER_3D(b, sizes,
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