#include "gpu_kernels.h"

template <>
struct TaylorSeriesFunctor<GPUDevice> {
	void operator()(
      const GPUDevice& d,
	  bool channels_first,
      int* size_data,
      int* size_coeffs,
	  int N,
      const float* input,
	  const float* coeffs,
      float* out){
		
		//get size and dimension information
		int n_b = size_data[0];
		int n_c = size_coeffs[0];
		int n_i = size_coeffs[1];
		int n_s = 1;
		for(int i = 0; i < N-1; i++)
			n_s *= size_data[i+1];
		n_s /= n_c;
		
		if(channels_first)
			taylor_series_channels_first(d, input, coeffs, out, n_b, n_s, n_c, n_i);
		else
			taylor_series_channels_last(d, input, coeffs, out, n_b, n_s, n_c, n_i);
	}
};


template <>
struct TaylorSeriesGradFunctor<GPUDevice> {
  void operator()(
      const GPUDevice& d,
	  bool channels_first,
      int* size_data,
      int* size_coeffs,
	  int N,
      const float* input,
	  const float* coeffs,
      const float* g,
      float* g_input,
      float* g_coeffs){
		
		//get size and dimension information
		int n_b = size_data[0];
		int n_c = size_coeffs[0];
		int n_i = size_coeffs[1];
		int n_s = 1;
		for(int i = 0; i < N-1; i++)
			n_s *= size_data[i+1];
		n_s /= n_c;
	}
};