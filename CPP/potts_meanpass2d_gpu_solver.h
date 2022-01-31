
#ifndef POTTS_MEANPASS_GPU_SOLVER_2D_
#define POTTS_MEANPASS_GPU_SOLVER_2D_

#include "potts_meanpass_gpu_solver.h"

class POTTS_MEANPASS_GPU_SOLVER_2D : public POTTS_MEANPASS_GPU_SOLVER_BASE
{
private:
    const int n_x;
    const int n_y;
    const float* const rx;
    const float* const ry;
	
protected:
    int min_iter_calc();
    void init_vars();
    void calculate_regularization();
    void parity_mask_buffer(float* buffer, const int parity);
    void parity_merge_buffer(float* buffer, const float* other, const int parity);
    void clean_up();
	
public:
	POTTS_MEANPASS_GPU_SOLVER_2D(
        const cudaStream_t & dev,
        const int batch,
        const int n_c,
        const int sizes[2],
        const float* data_cost,
        const float* rx_cost,
        const float* ry_cost,
        const float* init_u,
        float* u,
		float** buffers_full
	);

    static inline int num_buffers_full(){ return 1; }
    static inline int num_buffers_images(){ return 0; }
};


class POTTS_MEANPASS_GPU_GRADIENT_2D : public POTTS_MEANPASS_GPU_GRADIENT_BASE
{
private:
    const int n_x;
    const int n_y;
    const float* const rx;
    const float* const ry;
    float* const g_rx;
    float* const g_ry;
	
protected:
    int min_iter_calc();
    void init_vars();
	void get_reg_gradients_and_push(float tau);
    void clean_up();

public:
	POTTS_MEANPASS_GPU_GRADIENT_2D(
        const cudaStream_t & dev,
        const int batch,
        const int n_c,
        const int sizes[2],
        const float* u,
        const float* g,
        const float* rx_cost,
        const float* ry_cost,
        float* g_d,
        float* g_rx,
        float* g_ry,
		float** full_buffs
	);

    static inline int num_buffers_full(){ return 3; }
    static inline int num_buffers_images(){ return 0; }
};

#endif
