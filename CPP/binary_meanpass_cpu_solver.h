    

#ifndef BINARY_MEANPASS_CPU_SOLVER_H
#define BINARY_MEANPASS_CPU_SOLVER_H

class BINARY_MEANPASS_CPU_SOLVER_BASE
{
private:

protected:
    const float channels_first;
    const int b;
    const int n_c;
    const int n_s;
    const float* const data;
    float* const u;
    float* r_eff;
    
    virtual int min_iter_calc() = 0;
    virtual void init_vars() = 0;
    virtual void calculate_regularization() = 0;
    virtual void parity_mask_buffer(float* buffer, const int parity) = 0;
    virtual void parity_merge_buffer(float* buffer, const float* other, const int parity) = 0;
    virtual void clean_up() = 0;
    float block_iter(int, bool);
    
    // optimization constants
	const float beta = 0.001f;
	const float epsilon = 0.0001f;
	const float tau = 0.5f;
    
public:
	
    BINARY_MEANPASS_CPU_SOLVER_BASE(
        const float channels_first,
        const int batch,
        const int n_s,
        const int n_c,
        const float* data_cost,
        const float* init_u,
        float* u) ;
        
    ~BINARY_MEANPASS_CPU_SOLVER_BASE();
    
    void operator()();
};


class BINARY_MEANPASS_CPU_GRADIENT_BASE
{
private:

protected:
    const float channels_first;
    const int b;
    const int n_c;
    const int n_s;
    const float* const logits;
    const float* const grad;
    float* const g_data;
    float* u;
	float* d_y;
	float* g_u;
    
    virtual int min_iter_calc() = 0;
    virtual void init_vars() = 0;
    virtual void get_reg_gradients_and_push(float tau) = 0;
    virtual void clean_up() = 0;
    void block_iter();
    
    // optimization constants
	const float beta = 0.0001f;
	const float epsilon = 0.01f;
	const float tau = 0.25f;
    
public:
	
    BINARY_MEANPASS_CPU_GRADIENT_BASE(
        const float channels_first,
		const int batch,
		const int n_s,
		const int n_c,
		const float* u,
		const float* g,
		float* g_d );
        
    ~BINARY_MEANPASS_CPU_GRADIENT_BASE();
    
    void operator()();
};

#endif //BINARY_MEANPASS_CPU_SOLVER_H
