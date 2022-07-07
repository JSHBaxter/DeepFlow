#ifndef SPATIAL_AUGLAG_CPU_SOLVER_H
#define SPATIAL_AUGLAG_CPU_SOLVER_H

class SPATIAL_AUGLAG_CPU_SOLVER_BASE
{
protected:
    const bool channels_first;
    const int n_channels;
    const int img_size;
    
    float* const g;
    float* const div;
    float* const p;

public:
	SPATIAL_AUGLAG_CPU_SOLVER_BASE(
        const bool channels_first,
        const int n_channels,
        const int img_size
	);
    
    static SPATIAL_AUGLAG_CPU_SOLVER_BASE* factory(const bool channels_first, const int n_c, const int dim, const int* const dims,
                                                   float* const g, float* const div, const float* const* r);
    
    virtual int get_min_iter() = 0;
    virtual int get_max_iter() = 0;
    
    virtual static int get_number_buffer_full() = 0;
    
    virtual void init() = 0;
    virtual void run() = 0;
    virtual void deinit() = 0;
};

#endif
