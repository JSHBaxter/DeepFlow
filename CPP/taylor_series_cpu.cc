 
template <>
struct TaylorSeriesFunctor<CPUDevice> {
	void operator()(
      const CPUDevice& d,
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
		
		//iterate through image
		if(channels_first)
			for(int b = 0, p = 0; b < n_b; b++)
			for(int c = 0; c < n_c; c++)
			for(int s = 0; s < n_s; s++, p++){
				float x = 1.0f;
				out[p] = coeffs[c*n_i+0];
				for(int i = 1; i < n_i; i++){
					x *= input[p] / (float) i;
					out[p] += coeffs[c*n_i+i]*x;
				}
			}
		else
			for(int b = 0, p = 0; b < n_b; b++)
			for(int s = 0; s < n_s; s++)
			for(int c = 0; c < n_c; c++, p++){
				float x = 1.0f;
				out[p] = coeffs[c*n_i+0];
				for(int i = 1; i < n_i; i++){
					x *= input[p] / (float) i;
					out[p] += coeffs[c*n_i+i]*x;
				}
			}
	
	}
};


template <>
struct TaylorSeriesGradFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d,
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
		
		//clear buffer for coeffs gradient
		for(int p = 0; p < n_c*n_i; p++)
			g_coeffs[p] = 0.0f;
			
		//iterate through image
		if(channels_first)
			for(int b = 0, p = 0; b < n_b; b++)
			for(int c = 0; c < n_c; c++)
			for(int s = 0; s < n_s; s++, p++){
				float x = 1.0f;
				g_input[p] = 0.0f;
				g_coeffs[c*n_i+0] += g[p];
				for(int i = 1; i < n_i; i++){
					g_input[p] += coeffs[c*n_i+i]*x;
					x *= input[p] / (float) i;
					g_coeffs[c*n_i+i] += x * g[p];
				}
				g_input[p] *= g[p];
			}
		else
			for(int b = 0, p = 0; b < n_b; b++)
			for(int s = 0; s < n_s; s++)
			for(int c = 0; c < n_c; c++, p++){
				float x = 1.0f;
				g_input[p] = 0.0f;
				g_coeffs[c*n_i+0] += g[p];
				for(int i = 1; i < n_i; i++){
					g_input[p] += coeffs[c*n_i+i]*x;
					x *= input[p] / (float) i;
					g_coeffs[c*n_i+i] += x * g[p];
				}
				g_input[p] *= g[p];
			}
	}
};