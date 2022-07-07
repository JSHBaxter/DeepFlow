#include "cpu_kernels.h"
#include <limits>
#include <iostream>
#define epsilon 0.00001f

#include <cmath>
void clear_buffer(const CPU_DEVICE & dev, float* buffer, const int n_s){
    for(int i = 0; i < n_s; i++)
        buffer[i] = 0.0f;
}
void clear_buffer(const CPU_DEVICE & dev, float* buffer1, float* buffer2, const int n_s){
    for(int i = 0; i < n_s; i++)
        buffer1[i] = buffer2[i] = 0.0f;
}
void clear_buffer(const CPU_DEVICE & dev, float* buffer1, float* buffer2, float* buffer3, const int n_s){
    for(int i = 0; i < n_s; i++)
        buffer1[i] = buffer2[i] = buffer3[i] = 0.0f;
}

void print_buffer(const CPU_DEVICE & dev, const float* buffer, const int n_s){
	for(int i = 0; i < n_s; i++)
		printf("%f ",buffer[i]);
	printf("\n");
}
void set_buffer(const CPU_DEVICE & dev, float* buffer, const float number, const int n_s){
    for(int i = 0; i < n_s; i++)
        buffer[i] = number;
}

void copy_buffer(const CPU_DEVICE & dev, const float* bufferin, float* bufferout, const int n_s){
    for(int i = 0; i < n_s; i++)
        bufferout[i] = bufferin[i];
}


void add_then_store(const CPU_DEVICE& dev, const float* addend1, const float* addend2, float* sum, const int size){
    for(int i = 0; i < size; i++){
        float res = addend1[i]+addend2[i];
        sum[i] = res;
    }
}

void add_then_store(const CPU_DEVICE& dev, const float* addend1, const float* addend2, float* sum1, float* sum2, const int size){
    for(int i = 0; i < size; i++){
        float res = addend1[i]+addend2[i];
        sum1[i] = res;
        sum2[i] = res;
    }
}



void inc_mult_buffer(const CPU_DEVICE& dev, const float* inc, float* acc, const int n_s, const float multi){
    for(int i = 0; i < n_s; i++)
        acc[i] += multi * inc[i];
}

void inc_buffer(const CPU_DEVICE & dev, const float* inc, float* acc, const int n_s){
    for(int i = 0; i < n_s; i++)
        acc[i] += inc[i];
}

void inc_buffer(const CPU_DEVICE & dev, const float inc, float* acc, const int n_s){
    for(int i = 0; i < n_s; i++)
        acc[i] += inc;
}

void ninc_buffer(const CPU_DEVICE & dev, const float* inc, float* acc, const int n_s){
    for(int i = 0; i < n_s; i++)
        acc[i] -= inc[i];
}

void inc_inc_minc_buffer(const CPU_DEVICE& dev, const float* inc1, const float* inc2, const float* minc, const float multi, float* acc, const int n_s){
    for(int i = 0; i < n_s; i++)
        acc[i] += inc1[i] + inc2[i] + multi*minc[i];
}

void m_inc_inc_ninc_minc_buffer(const CPU_DEVICE& dev, const float* inc1, const float* inc2, const float* ninc, const float* minc, const float multi_end, const float multi_all, float* acc, const int n_s){
    for(int i = 0; i < n_s; i++)
        acc[i] += multi_all*(inc1[i]+inc2[i]-ninc[i]+multi_end*minc[i]);
}

void log_buffer(const CPU_DEVICE & dev, const float* bufferin, float* bufferout, const int n_s){
    for(int i = 0; i < n_s; i++)
		if( bufferin[i] < epsilon )
			bufferout[i] = log(epsilon);
		else
			bufferout[i] = log(bufferin[i]);
}

void div_buffer(const CPU_DEVICE & dev, const float number, float* buffer, const int n_s){
    mult_buffer(dev, 1.0f/number, buffer, n_s);
}

void mult_buffer(const CPU_DEVICE & dev, const float number, float* buffer, const int n_s){
    for(int i = 0; i < n_s; i++)
        buffer[i] *= number;
}

void constrain(const CPU_DEVICE & dev, float* buffer, const float* constraint, const int n_s){
    for(int i = 0; i < n_s; i++)
        if(buffer[i] > constraint[i])
            buffer[i] = constraint[i];
}

void max_neg_constrain(const CPU_DEVICE& dev, float* buffer, const float* constraint, const int n_s){
    for(int i = 0; i < n_s; i++)
        if(buffer[i] > -constraint[i])
            buffer[i] = -constraint[i];
}

float max_of_buffer(const CPU_DEVICE & dev, const float* buffer, const int n_s){
    float maxabs = 0.0f;
    for(int i = 0; i < n_s; i++){
        if( buffer[i] < -maxabs )
            maxabs = -buffer[i];
        else if( buffer[i] > maxabs )
            maxabs = buffer[i];
    }
    return maxabs;
}

float meanabs(const CPU_DEVICE & dev, const float* buffer, const int n_s){
    double sumabs = 0.0f;
    for(int i = 0; i < n_s; i++)
        sumabs += (buffer[i] < 0) ? -buffer[i] : buffer[i];
    return (float) (sumabs / (double) n_s);
}

float spatmaxabs_dev_channels_first(const float* buffer, const int n_s, const int n_c){
    double maxabs = 0.0;
    float sumabs = 0.0f;
    for(int s = 0; s < n_s; s++){
        sumabs = 0.0f;
        for(int c = 0; c < n_c; c++)
            sumabs += (buffer[c*n_s+s] < 0) ? -buffer[c*n_s+s] : buffer[c*n_s+s];
        maxabs = (maxabs > sumabs) ? maxabs : sumabs;
    }
    return maxabs;
}

float spatmaxabs_dev_channels_last(const float* buffer, const int n_s, const int n_c){
    double maxabs = 0.0;
    float sumabs = 0.0f;
    for(int s = 0; s < n_s; s++){
        sumabs = 0.0f;
        for(int c = 0; c < n_c; c++)
            sumabs += (buffer[s*n_c+c] < 0) ? -buffer[s*n_c+c] : buffer[s*n_c+c];
        maxabs = (maxabs > sumabs) ? maxabs : sumabs;
    }
    return maxabs;
}

float spat_max_of_buffer(const CPU_DEVICE& dev, const float* buffer, const int n_s, const int n_c){
    if(dev.channels_first)
        return spatmaxabs_dev_channels_first(buffer, n_s, n_c);
    else
        return spatmaxabs_dev_channels_last(buffer, n_s, n_c);
}

float max_diff(const CPU_DEVICE & dev, const float* buffer, const int n_c, const int n_s){
    float avg_all_diff = 0.0;
    for(int s = 0; s < n_s; s++){
        float max_diff = 0.0;
        for(int c1 = 0; c1 < n_c; c1++)
        for(int c2 = 0; c2 < n_c; c2++)
            if( buffer[s*n_c+c1]-buffer[s*n_c+c2] > max_diff )
                max_diff = buffer[s*n_c+c1]-buffer[s*n_c+c2];
        avg_all_diff += max_diff;
    }
    avg_all_diff /= (float) n_s;
    return avg_all_diff;
}

float* transpose(const CPU_DEVICE & dev, const float* bufferin, float* bufferout, const int n_d1, const int n_d2){
    for(int d1 = 0; d1 < n_d1; d1++)
        for(int d2 = 0; d2 < n_d2; d2++)
            bufferout[d2*n_d1+d1] = bufferin[d1*n_d2+d2];
    return bufferout;
}

void unfold_buffer(const CPU_DEVICE & dev, float* buffer, const int n_s, const int n_c, const int n_r){
	for(int s = n_s-1; s >= 0; s--){
		for(int c = n_c-1; c >= 0; c--)
			buffer[n_r*s+c] = buffer[n_c*s+c];
		for(int c = n_c; c < n_r; c++)
			buffer[n_r*s+c] = 0.0f;
	}
}

void refold_buffer(const CPU_DEVICE & dev, float* buffer, const int n_s, const int n_c, const int n_r){
	for(int s = 0; s < n_s; s++)
	for(int c = 0; c < n_c; c++)
		buffer[n_c*s+c] = buffer[n_r*s+c];
}

inline int idx(const int x, const int n_x, const int y, const int n_y){
    return y + n_y*x;
}

inline int idx(const int x, const int n_x, const int y, const int n_y, const int z, const int n_z){
    return z + n_z*idx(x,n_x,y,n_y);
}

inline int idx(const int x, const int n_x, const int y, const int n_y, const int z, const int n_z, const int w, const int n_w){
    return w + n_w*idx(x,n_x,y,n_y,z,n_z);
}

inline int idxc(const int s, const int n_s, const int c, const int n_c){
    return c + n_c*s;
}

inline int idxc(const int x, const int n_x, const int y, const int n_y, const int c, const int n_c){
    return c + n_c*idx(x,n_x,y,n_y);
}

inline int idxc(const int x, const int n_x, const int y, const int n_y, const int z, const int n_z, const int c, const int n_c){
    return c + n_c*idx(x,n_x,y,n_y,z,n_z);
}

void softmax_channels_first(const float* bufferin, float* bufferout, const int n_s, const int n_c){
    for(int s = 0; s < n_s; s++) {
        float max_cost = bufferin[s];
        for(int c = 1; c < n_c; c++)
            if(bufferin[n_s* c + s] > max_cost)
                max_cost = bufferin[n_s* c + s];
        float accum = 0.0f;
        for(int c = 0; c < n_c; c++){
            bufferout[n_s* c + s] = std::exp(bufferin[n_s* c + s]-max_cost);
            accum += bufferout[n_s* c + s];
        }
        for(int c = 0; c < n_c; c++)
            bufferout[n_s* c + s] /= accum;
    }
}

void softmax_channels_first(const float* bufferin1, const float* bufferin2, float* bufferout, const int n_s, const int n_c){
    for(int s = 0; s < n_s; s++) {
        float max_cost = bufferin1[s]+bufferin2[s];
        for(int c = 1; c < n_c; c++)
            if(bufferin1[n_s* c + s]+bufferin2[n_s* c + s] > max_cost)
                max_cost = bufferin1[n_s* c + s]+bufferin2[n_s* c + s];
        float accum = 0.0f;
        for(int c = 0; c < n_c; c++){
            bufferout[n_s* c + s] = std::exp(bufferin1[n_s* c + s]+bufferin2[n_s* c + s]-max_cost);
            accum += bufferout[n_s* c + s];
        }
        for(int c = 0; c < n_c; c++)
            bufferout[n_s* c + s] /= accum;
    }
}

void softmax_channels_last(const float* bufferin, float* bufferout, const int n_s, const int n_c){
    for(int s = 0; s < n_s; s++) {
        float max_cost = bufferin[n_c*s];
        for(int c = 1; c < n_c; c++)
            if(bufferin[c + n_c*s] > max_cost)
                max_cost = bufferin[c + n_c*s];
        float accum = 0.0f;
        for(int c = 0; c < n_c; c++){
            bufferout[c + n_c*s] = std::exp(bufferin[c + n_c*s]-max_cost);
            accum += bufferout[c + n_c*s];
        }
        for(int c = 0; c < n_c; c++)
            bufferout[c + n_c*s] /= accum;
    }
}

void softmax_channels_last(const float* bufferin1, const float* bufferin2, float* bufferout, const int n_s, const int n_c){
    for(int s = 0; s < n_s; s++) {
        float max_cost = bufferin1[n_c*s]+bufferin2[n_c*s];
        for(int c = 1; c < n_c; c++)
            if(bufferin1[c + n_c*s]+bufferin2[c + n_c*s] > max_cost)
                max_cost = bufferin1[c + n_c*s]+bufferin2[c + n_c*s];
        float accum = 0.0f;
        for(int c = 0; c < n_c; c++){
            bufferout[c + n_c*s] = std::exp(bufferin1[c + n_c*s]+bufferin2[c + n_c*s]-max_cost);
            accum += bufferout[c + n_c*s];
        }
        for(int c = 0; c < n_c; c++)
            bufferout[c + n_c*s] /= accum;
    }
}

void softmax(const CPU_DEVICE & dev, const float* bufferin, float* bufferout, const int n_s, const int n_c){
    if(dev.channels_first)
        softmax_channels_first(bufferin,bufferout,n_s,n_c);
    else
        softmax_channels_last(bufferin,bufferout,n_s,n_c);
}

void softmax(const CPU_DEVICE & dev, const float* bufferin1, const float* bufferin2, float* bufferout, const int n_s, const int n_c){
    if(!bufferin2)
        softmax(dev,bufferin1,bufferout,n_s,n_c);
    else if(dev.channels_first)
        softmax_channels_first(bufferin1,bufferin2,bufferout,n_s,n_c);
    else
        softmax_channels_last(bufferin1,bufferin2,bufferout,n_s,n_c);
}

void sigmoid(const CPU_DEVICE & dev, const float* bufferin, float* bufferout, const int n_s){
    for(int s = 0; s < n_s; s++) {
		float cost = bufferin[s];
        bufferout[s] = 1.0f / (1.0f + std::exp(-cost));
    }
}


void sigmoid(const CPU_DEVICE & dev,const float* bufferin1, const float* bufferin2, float* bufferout, const int n_s){
    if(! bufferin2 ){
        sigmoid(dev, bufferin1, bufferout, n_s);
        return;
    }
    for(int s = 0; s < n_s; s++) {
		float cost = bufferin1[s] + bufferin2[s];
        bufferout[s] = 1.0f / (1.0f + std::exp(-cost));
    }
}

void exp(const CPU_DEVICE & dev, const float* bufferin, float* bufferout, const int n_s){
    for(int s = 0; s < n_s; s++) {
		float cost = bufferin[s];
        bufferout[s] = std::exp(cost);
    }
}

void parity_mask_channels_last(float* const buffer, const int n_x, const int n_c, const int parity){
	for(int x = 0, s = 0; x < n_x; x++)
        for(int c = 0; c < n_c; c++, s++)
            buffer[s] *= (parity ^ x) & 1;
}

void parity_mask_channels_last(float* const buffer, const int n_x, const int n_y, const int n_c, const int parity){
	for(int x = 0, s = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
        for(int c = 0; c < n_c; c++, s++)
            buffer[s] *= (parity ^ x ^ y) & 1;
}

void parity_mask_channels_last(float* const buffer, const int n_x, const int n_y, const int n_z, const int n_c, const int parity){
	for(int x = 0, s = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int z = 0; z < n_z; z++)
        for(int c = 0; c < n_c; c++, s++)
            buffer[s] *= (parity ^ x ^ y ^ z) & 1;
}

void parity_mask_channels_first(float* const buffer, const int n_x, const int n_c, const int parity){
	for(int c = 0, s = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++, s++)
        buffer[s] *= (parity ^ x) & 1;
}

void parity_mask_channels_first(float* const buffer, const int n_x, const int n_y, const int n_c, const int parity){
	for(int c = 0,  s = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++, s++)
            buffer[s] *= (parity ^ x ^ y) & 1;
}

void parity_mask_channels_first(float* const buffer, const int n_x, const int n_y, const int n_z, const int n_c, const int parity){
	for(int c = 0, s = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int z = 0; z < n_z; z++, s++)
            buffer[s] *= (parity ^ x ^ y ^ z) & 1;
}

void parity_mask(const CPU_DEVICE & dev, float* const buffer, const int dim, const int* const n, const int n_c, const int parity){
    if(dev.channels_first)
        switch(dim){
            case 1:
                parity_mask_channels_first(buffer, n[0], n_c, parity);
                break;
            case 2:
                parity_mask_channels_first(buffer, n[0], n[1], n_c, parity);
                break;
            case 3:
                parity_mask_channels_first(buffer, n[0], n[1], n[2], n_c, parity);
                break;
        }
    else
        switch(dim){
            case 1:
                parity_mask_channels_last(buffer, n[0], n_c, parity);
                break;
            case 2:
                parity_mask_channels_last(buffer, n[0], n[1], n_c, parity);
                break;
            case 3:
                parity_mask_channels_last(buffer, n[0], n[1], n[2], n_c, parity);
                break;
        }
}

void parity_mask_channels_last(float* const buffer, const float* other, const int n_x, const int n_c, const int parity){
	for(int x = 0, s = 0; x < n_x; x++)
        for(int c = 0; c < n_c; c++, s++){
            buffer[s] *= (parity ^ x) & 1;
            buffer[s] += ((parity ^ x ^ 1) & 1) * other[s];
        }
}

void parity_mask_channels_last(float* const buffer, const float* other, const int n_x, const int n_y, const int n_c, const int parity){
	for(int x = 0, s = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
        for(int c = 0; c < n_c; c++, s++){
            buffer[s] *= (parity ^ x ^ y) & 1;
            buffer[s] += ((parity ^ x ^ y ^ 1) & 1) * other[s];
        }
}

void parity_mask_channels_last(float* const buffer, const float* other, const int n_x, const int n_y, const int n_z, const int n_c, const int parity){
	for(int x = 0, s = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int z = 0; z < n_z; z++)
        for(int c = 0; c < n_c; c++, s++){
            buffer[s] *= (parity ^ x ^ y ^ z) & 1;
            buffer[s] += ((parity ^ x ^ y ^ z ^ 1) & 1) * other[s];
        }
}

void parity_mask_channels_first(float* const buffer, const float* other, const int n_x, const int n_c, const int parity){
	for(int c = 0, s = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++, s++){
            buffer[s] *= (parity ^ x) & 1;
            buffer[s] += ((parity ^ x ^ 1) & 1) * other[s];
        }
}

void parity_mask_channels_first(float* const buffer, const float* other, const int n_x, const int n_y, const int n_c, const int parity){
	for(int c = 0, s = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++, s++){
            buffer[s] *= (parity ^ x ^ y) & 1;
            buffer[s] += ((parity ^ x ^ y ^ 1) & 1) * other[s];
        }
}

void parity_mask_channels_first(float* const buffer, const float* other, const int n_x, const int n_y, const int n_z, const int n_c, const int parity){
	for(int c = 0, s = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int z = 0; z < n_z; z++, s++){
            buffer[s] *= (parity ^ x ^ y ^ z) & 1;
            buffer[s] += ((parity ^ x ^ y ^ z ^ 1) & 1) * other[s];
        }
}

void parity_mask(const CPU_DEVICE & dev, float* const buffer, const float* other, const int dim, const int* n, const int n_c, const int parity){
    if(dev.channels_first)
        switch(dim){
            case 1:
                parity_mask_channels_first(buffer, other, n[0], n_c, parity);
                break;
            case 2:
                parity_mask_channels_first(buffer, other, n[0], n[1], n_c, parity);
                break;
            case 3:
                parity_mask_channels_first(buffer, other, n[0], n[1], n[2], n_c, parity);
                break;
        }
    else
        switch(dim){
            case 1:
                parity_mask_channels_last(buffer, other, n[0], n_c, parity);
                break;
            case 2:
                parity_mask_channels_last(buffer, other, n[0], n[1], n_c, parity);
                break;
            case 3:
                parity_mask_channels_last(buffer, other, n[0], n[1], n[2], n_c, parity);
                break;
        }
}

void change_to_diff(const CPU_DEVICE & dev, float* buffer, float* update, const int n_s, const float alpha){
	for(int s = 0; s < n_s; s++) {
		float diff = alpha * (update[s]-buffer[s]);
		buffer[s] += diff;
        update[s] = diff;
	}
}

void calculate_r_eff_channels_last(float* r_eff, const float* rx, const float* ry, const float* rz, const float* u, const int n_x, const int n_y, const int n_z, const int n_c) {
    
    for (int x = 0; x < n_x; x++)
    for (int y = 0; y < n_y; y++) 
    for (int z = 0; z < n_z; z++)
    for (int c = 0; c < n_c; c++) {
        r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] = 0.0f;

        //in z+
        if(z < n_z-1)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += rz[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] * (2.0*u[idxc(x,n_x,y,n_y,z+1,n_z,c,n_c)]-1.0);

        //in z-
        if(z > 0)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += rz[idxc(x,n_x,y,n_y,z-1,n_z,c,n_c)] * (2.0*u[idxc(x,n_x,y,n_y,z-1,n_z,c,n_c)]-1.0);

        //in y+
        if(y < n_y-1)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += ry[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] * (2.0*u[idxc(x,n_x,y+1,n_y,z,n_z,c,n_c)]-1.0);

        //in y-
        if(y > 0)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += ry[idxc(x,n_x,y-1,n_y,z,n_z,c,n_c)] * (2.0*u[idxc(x,n_x,y-1,n_y,z,n_z,c,n_c)]-1.0);

        //in x+
        if(x < n_x-1)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += rx[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] * (2.0*u[idxc(x+1,n_x,y,n_y,z,n_z,c,n_c)]-1.0);

        //in x-
        if(x > 0)
            r_eff[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += rx[idxc(x-1,n_x,y,n_y,z,n_z,c,n_c)] * (2.0*u[idxc(x-1,n_x,y,n_y,z,n_z,c,n_c)]-1.0);

    }
}

void calculate_r_eff_channels_last(float* r_eff, const float* rx, const float* ry, const float* u, const int n_x, const int n_y, const int n_c) {
    
    for (int x = 0; x < n_x; x++)
    for (int y = 0; y < n_y; y++) 
    for (int c = 0; c < n_c; c++) {
        r_eff[idxc(x,n_x,y,n_y,c,n_c)] = 0.0f;
        
        //in y+
        if(y < n_y-1)
            r_eff[idxc(x,n_x,y,n_y,c,n_c)] += ry[idxc(x,n_x,y,n_y,c,n_c)] * (2.0*u[idxc(x,n_x,y+1,n_y,c,n_c)]-1.0);

        //in y-
        if(y > 0)
            r_eff[idxc(x,n_x,y,n_y,c,n_c)] += ry[idxc(x,n_x,y-1,n_y,c,n_c)] * (2.0*u[idxc(x,n_x,y-1,n_y,c,n_c)]-1.0);

        //in x+
        if(x < n_x-1)
            r_eff[idxc(x,n_x,y,n_y,c,n_c)] += rx[idxc(x,n_x,y,n_y,c,n_c)] * (2.0*u[idxc(x+1,n_x,y,n_y,c,n_c)]-1.0);

        //in x-
        if(x > 0)
            r_eff[idxc(x,n_x,y,n_y,c,n_c)] += rx[idxc(x-1,n_x,y,n_y,c,n_c)] * (2.0*u[idxc(x-1,n_x,y,n_y,c,n_c)]-1.0);

    }
}

void calculate_r_eff_channels_last(float* r_eff, const float* rx, const float* u, const int n_x, const int n_c) {
    
    for (int x = 0; x < n_x; x++)
    for (int c = 0; c < n_c; c++) {
        r_eff[idxc(x,n_x,c,n_c)] = 0.0f;

        //in x+
        if(x < n_x-1)
            r_eff[idxc(x,n_x,c,n_c)] += rx[idxc(x,n_x,c,n_c)] * (2.0*u[idxc(x+1,n_x,c,n_c)]-1.0);

        //in x-
        if(x > 0)
            r_eff[idxc(x,n_x,c,n_c)] += rx[idxc(x-1,n_x,c,n_c)] * (2.0*u[idxc(x-1,n_x,c,n_c)]-1.0);

    }
}

void calculate_r_eff_channels_first(float* r_eff, const float* rx, const float* ry, const float* rz, const float* u, const int n_x, const int n_y, const int n_z, const int n_c) {
    
    for (int c = 0; c < n_c; c++)
    for (int x = 0; x < n_x; x++)
    for (int y = 0; y < n_y; y++) 
    for (int z = 0; z < n_z; z++) {
        r_eff[idx(c,n_c,x,n_x,y,n_y,z,n_z)] = 0.0f;

        //in z+
        if(z < n_z-1)
            r_eff[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += rz[idx(c,n_c,x,n_x,y,n_y,z  ,n_z)] * (2.0*u[idx(c,n_c,x,n_x,y,n_y,z+1,n_z)]-1.0);

        //in z-
        if(z > 0)
            r_eff[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += rz[idx(c,n_c,x,n_x,y,n_y,z-1,n_z)] * (2.0*u[idx(c,n_c,x,n_x,y,n_y,z-1,n_z)]-1.0);

        //in y+
        if(y < n_y-1)
            r_eff[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += ry[idx(c,n_c,x,n_x,y  ,n_y,z,n_z)] * (2.0*u[idx(c,n_c,x,n_x,y+1,n_y,z,n_z)]-1.0);

        //in y-
        if(y > 0)
            r_eff[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += ry[idx(c,n_c,x,n_x,y-1,n_y,z,n_z)] * (2.0*u[idx(c,n_c,x,n_x,y-1,n_y,z,n_z)]-1.0);

        //in x+
        if(x < n_x-1)
            r_eff[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += rx[idx(c,n_c,x  ,n_x,y,n_y,z,n_z)] * (2.0*u[idx(c,n_c,x+1,n_x,y,n_y,z,n_z)]-1.0);

        //in x-
        if(x > 0)
            r_eff[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += rx[idx(c,n_c,x-1,n_x,y,n_y,z,n_z)] * (2.0*u[idx(c,n_c,x-1,n_x,y,n_y,z,n_z)]-1.0);

    }
}

void calculate_r_eff_channels_first(float* r_eff, const float* rx, const float* ry, const float* u, const int n_x, const int n_y, const int n_c) {
    
    for (int c = 0; c < n_c; c++) 
    for (int x = 0; x < n_x; x++)
    for (int y = 0; y < n_y; y++) {
        r_eff[idx(c,n_c,x,n_x,y,n_y)] = 0.0f;
        
        //in y+
        if(y < n_y-1)
            r_eff[idx(c,n_c,x,n_x,y,n_y)] += ry[idx(c,n_c,x,n_x,y  ,n_y)] * (2.0*u[idx(c,n_c,x,n_x,y+1,n_y)]-1.0);

        //in y-
        if(y > 0)
            r_eff[idx(c,n_c,x,n_x,y,n_y)] += ry[idx(c,n_c,x,n_x,y-1,n_y)] * (2.0*u[idx(c,n_c,x,n_x,y-1,n_y)]-1.0);

        //in x+
        if(x < n_x-1)
            r_eff[idx(c,n_c,x,n_x,y,n_y)] += rx[idx(c,n_c,x  ,n_x,y,n_y)] * (2.0*u[idx(c,n_c,x+1,n_x,y,n_y)]-1.0);

        //in x-
        if(x > 0)
            r_eff[idx(c,n_c,x,n_x,y,n_y)] += rx[idx(c,n_c,x-1,n_x,y,n_y)] * (2.0*u[idx(c,n_c,x-1,n_x,y,n_y)]-1.0);

    }
}

void calculate_r_eff_channels_first(float* r_eff, const float* rx, const float* u, const int n_x, const int n_c) {
    
    for (int c = 0; c < n_c; c++) 
    for (int x = 0; x < n_x; x++) {
        r_eff[idx(c,n_c,x,n_x)] = 0.0f;

        //in x+
        if(x < n_x-1)
            r_eff[idx(c,n_c,x,n_x)] += rx[idx(c,n_c,x,n_x)] * (2.0*u[idx(c,n_c,x+1,n_x)]-1.0);

        //in x-
        if(x > 0)
            r_eff[idx(c,n_c,x,n_x)] += rx[idx(c,n_c,x-1,n_x)] * (2.0*u[idx(c,n_c,x-1,n_x)]-1.0);

    }
}

void get_effective_reg(const CPU_DEVICE & dev, float* const r_eff, const float* const u, const float *const *const r, const int dim, const int* const n, const int n_c){
    if(dev.channels_first)
        switch(dim){
            case 1:
                calculate_r_eff_channels_first(r_eff, r[0], u, n[0], n_c);
                break;
            case 2:
                calculate_r_eff_channels_first(r_eff, r[0], r[1], u, n[0], n[1], n_c);
                break;
            case 3:
                calculate_r_eff_channels_first(r_eff, r[0], r[1], r[2], u, n[0], n[1], n[2], n_c);
                break;
        }
    else
        switch(dim){
            case 1:
                calculate_r_eff_channels_last(r_eff, r[0], u, n[0], n_c);
                break;
            case 2:
                calculate_r_eff_channels_last(r_eff, r[0], r[1], u, n[0], n[1], n_c);
                break;
            case 3:
                calculate_r_eff_channels_last(r_eff, r[0], r[1], r[2], u, n[0], n[1], n[2], n_c);
                break;
        }
}

void aggregate_bottom_up_channels_first(float* buffer, const int n_s, const int n_r, const DAGNode* const* bottom_up_list){
    
    for (int l = 0; l < n_r; l++) {
        const DAGNode* n = bottom_up_list[l];
        if(n->d == -1)
            for(int c = 0; c < n->c; c++)
            for (int s = 0; s < n_s; s++)
                buffer[idx(n->r,n_r,s,n_s)] += n->child_weight[c]*buffer[idx(n->children[c]->r,n_r,s,n_s)];
        }
}

void aggregate_bottom_up_channels_first(const float* bufferin, float* bufferout, const int n_s, const int n_c, const int n_r, const DAGNode* const* bottom_up_list){
    
    for (int l = 0; l < n_r; l++) {
        const DAGNode* n = bottom_up_list[l];
        if(n->c > 0){
            for (int s = 0; s < n_s; s++)
                bufferout[idx(n->r,n_r,s,n_s)] = n->child_weight[0]*bufferout[idx(n->children[0]->r,n_r,s,n_s)];
            for(int c = 1; c < n->c; c++)
            for(int s = 0; s < n_s; s++)
                bufferout[idx(n->r,n_r,s,n_s)] += n->child_weight[c]*bufferout[idx(n->children[c]->r,n_r,s,n_s)];
        }else{
            for (int s = 0; s < n_s; s++)
                bufferout[idx(n->r,n_r,s,n_s)] = bufferin[idx(n->d,n_c,s,n_s)];
        }
    }
}

void aggregate_top_down_channels_first(float* buffer, const int n_s, const int n_r, const DAGNode* const* bottom_up_list){
	
    for (int l = n_r-1; l >= 0; l--) {
        const DAGNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
        for(int s = 0; s < n_s; s++)
            buffer[idx(n->children[c]->r,n_r,s,n_s)] += n->child_weight[c]*buffer[idx(n->r,n_r,s,n_s)];
    }
}

void aggregate_bottom_up_channels_first(float* buffer, const int n_s, const int n_r, const TreeNode* const* bottom_up_list){
    
    for (int l = 0; l < n_r; l++) {
        const TreeNode* n = bottom_up_list[l];
        if(n->d == -1)
            for(int c = 0; c < n->c; c++)
            for (int s = 0; s < n_s; s++)
                buffer[idx(n->r,n_r,s,n_s)] += buffer[idx(n->children[c]->r,n_r,s,n_s)];
        }
}

void aggregate_bottom_up_channels_first(const float* bufferin, float* bufferout, const int n_s, const int n_c, const int n_r, const TreeNode* const* bottom_up_list){
    
    for (int l = 0; l < n_r; l++) {
        const TreeNode* n = bottom_up_list[l];
        if(n->c > 0){
            for (int s = 0; s < n_s; s++)
                bufferout[idx(n->r,n_r,s,n_s)] = bufferout[idx(n->children[0]->r,n_r,s,n_s)];
            for(int c = 1; c < n->c; c++)
            for(int s = 0; s < n_s; s++)
                bufferout[idx(n->r,n_r,s,n_s)] += bufferout[idx(n->children[c]->r,n_r,s,n_s)];
        }else{
            for (int s = 0; s < n_s; s++)
                bufferout[idx(n->r,n_r,s,n_s)] = bufferin[idx(n->d,n_c,s,n_s)];
        }
    }
}

void aggregate_top_down_channels_first(float* buffer, const int n_s, const int n_r, const TreeNode* const* bottom_up_list){
	
    for (int l = n_r-1; l >= 0; l--) {
        const TreeNode* n = bottom_up_list[l];
        for(int c = 0; c < n->c; c++)
        for(int s = 0; s < n_s; s++)
            buffer[idx(n->children[c]->r,n_r,s,n_s)] += buffer[idx(n->r,n_r,s,n_s)];
    }
}

void aggregate_bottom_up(float* buffer, const int n_s, const int n_r, const TreeNode* const* bottom_up_list){
    for (int s = 0; s < n_s; s++)
        for (int l = 0; l < n_r; l++) {
            const TreeNode* n = bottom_up_list[l];
            if(n->d == -1){
                for(int c = 0; c < n->c; c++)
                    buffer[idxc(s,n_s,n->r,n_r)] += buffer[idxc(s,n_s,n->children[c]->r,n_r)];
            }else{
                buffer[idxc(s,n_s,n->r,n_r)] = buffer[idxc(s,n_s,n->d,n_r)];
            }
        }
}

void aggregate_bottom_up(const float* bufferin, float* bufferout, const int n_s, const int n_c, const int n_r, const TreeNode* const* bottom_up_list){
    for (int s = 0; s < n_s; s++)
        for (int l = 0; l < n_r; l++) {
            const TreeNode* n = bottom_up_list[l];
            if(n->c > 0){
                bufferout[idxc(s,n_s,n->r,n_r)] = 0.0f;
                for(int c = 0; c < n->c; c++)
                    bufferout[idxc(s,n_s,n->r,n_r)] += bufferout[idxc(s,n_s,n->children[c]->r,n_r)];
            }else{
                bufferout[idxc(s,n_s,n->r,n_r)] = bufferin[idxc(s,n_s,n->d,n_c)];
            }
        }
}

void aggregate_top_down(float* buffer, const int n_s, const int n_r, const TreeNode* const* bottom_up_list){
	for(int s = 0; s < n_s; s++)
        for (int l = n_r-1; l >= 0; l--) {
            const TreeNode* n = bottom_up_list[l];
			for(int c = 0; c < n->c; c++)
				buffer[idxc(s,n_s,n->children[c]->r,n_r)] += buffer[idxc(s,n_s,n->r,n_r)];
		}
}

void untangle_softmax_channels_first(const float* g, const float* u, float* dy, const int n_s, const int n_c){
	for(int s = 0; s < n_s; s++)
		for (int c = 0; c < n_c; c++){
			float new_grad = 0.0f;
			float uc = u[idx(c,n_c,s,n_s)];
			for(int a = 0; a < n_c; a++){
				float da = g[idx(a,n_c,s,n_s)];
				if(c == a)
					new_grad += da*(1.0f-uc);
				else
					new_grad -= da*u[idx(a,n_c,s,n_s)];
			}
			dy[idx(c,n_c,s,n_s)] = new_grad*uc;
		}
}

void untangle_softmax_channels_last(const float* g, const float* u, float* dy, const int n_s, const int n_c){
	for(int s = 0; s < n_s; s++)
		for (int c = 0; c < n_c; c++){
			float new_grad = 0.0f;
			float uc = u[idxc(s,n_s,c,n_c)];
			for(int a = 0; a < n_c; a++){
				float da = g[idxc(s,n_s,a,n_c)];
				if(c == a)
					new_grad += da*(1.0f-uc);
				else
					new_grad -= da*u[idxc(s,n_s,a,n_c)];
			}
			dy[idxc(s,n_s,c,n_c)] = new_grad*uc;
		}
}

void untangle_softmax(const CPU_DEVICE & dev, const float* g, const float* u, float* dy, const int n_s, const int n_c){
    if(dev.channels_first)
        untangle_softmax_channels_first(g,u,dy,n_s,n_c);
    else
        untangle_softmax_channels_last(g,u,dy,n_s,n_c);
}

void untangle_sigmoid(const CPU_DEVICE & dev, const float* g, const float* u, float* dy, const int n_s){
	for(int s = 0; s < n_s; s++)
		dy[s] = g[s]*u[s]*(1-u[s]);
}

void get_gradient_for_u_channels_last(const float* dy, const float* rx, const float* ry, const float* rz, float* du, const int n_x, const int n_y, const int n_z, const int n_c, const float tau){
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int z = 0; z < n_z; z++)
	for(int c = 0; c < n_c; c++){
		float grad_val = 0.0f;

		//z down
		if( z > 0 )
			grad_val += 2.0f*dy[idxc(x,n_x,y,n_y,z-1,n_z,c,n_c)]*rz[idxc(x,n_x,y,n_y,z-1,n_z,c,n_c)];

		//y down
		if( y > 0 )
			grad_val += 2.0f*dy[idxc(x,n_x,y-1,n_y,z,n_z,c,n_c)]*ry[idxc(x,n_x,y-1,n_y,z,n_z,c,n_c)];

		//x down
		if( x > 0 )
			grad_val += 2.0f*dy[idxc(x-1,n_x,y,n_y,z,n_z,c,n_c)]*rx[idxc(x-1,n_x,y,n_y,z,n_z,c,n_c)];

		//z up
		if( z < n_z - 1)
			grad_val += 2.0f*dy[idxc(x,n_x,y,n_y,z+1,n_z,c,n_c)]*rz[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];

		//y up
		if ( y < n_y - 1)
			grad_val += 2.0f*dy[idxc(x,n_x,y+1,n_y,z,n_z,c,n_c)]*ry[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];

		//x up
		if ( x < n_x - 1)
			grad_val += 2.0f*dy[idxc(x+1,n_x,y,n_y,z,n_z,c,n_c)]*rx[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];

		du[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] = tau*grad_val + (1.0f-tau)*du[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];
	}
}

void get_gradient_for_u_channels_last(const float* dy, const float* rx, const float* ry, float* du, const int n_x, const int n_y, const int n_c, const float tau){
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int c = 0; c < n_c; c++){
		float grad_val = 0.0f;
		//y down
		if( y > 0 )
			grad_val += 2.0f*dy[idxc(x,n_x,y-1,n_y,c,n_c)]*ry[idxc(x,n_x,y-1,n_y,c,n_c)];

		//x down
		if( x > 0 )
			grad_val += 2.0f*dy[idxc(x-1,n_x,y,n_y,c,n_c)]*rx[idxc(x-1,n_x,y,n_y,c,n_c)];
		
		//y up
		if ( y < n_y - 1)
			grad_val += 2.0f*dy[idxc(x,n_x,y+1,n_y,c,n_c)]*ry[idxc(x,n_x,y,n_y,c,n_c)];

		//x up
		if ( x < n_x - 1)
			grad_val += 2.0f*dy[idxc(x+1,n_x,y,n_y,c,n_c)]*rx[idxc(x,n_x,y,n_y,c,n_c)];

		du[idxc(x,n_x,y,n_y,c,n_c)] = tau*grad_val + (1.0f-tau)*du[idxc(x,n_x,y,n_y,c,n_c)];
	}
}

void get_gradient_for_u_channels_last(const float* dy, const float* rx, float* du, const int n_x, const int n_c, const float tau){
	for(int x = 0; x < n_x; x++)
	for(int c = 0; c < n_c; c++){
		float grad_val = 0.0f;
		//x down
		if( x > 0 )
			grad_val += 2.0f*dy[idxc(x-1,n_x,c,n_c)]*rx[idxc(x-1,n_x,c,n_c)];
		
		//x up
		if ( x < n_x - 1)
			grad_val += 2.0f*dy[idxc(x+1,n_x,c,n_c)]*rx[idxc(x,n_x,c,n_c)];

		du[idxc(x,n_x,c,n_c)] = tau*grad_val + (1.0f-tau)*du[idxc(x,n_x,c,n_c)];
	}
}

void get_gradient_for_u_channels_first(const float* dy, const float* rx, const float* ry, const float* rz, float* du, const int n_x, const int n_y, const int n_z, const int n_c, const float tau){
	for(int c = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int z = 0; z < n_z; z++){
		float grad_val = 0.0f;

		//z down
		if( z > 0 )
			grad_val += 2.0f*dy[idx(c,n_c,x,n_x,y,n_y,z-1,n_z)]*rz[idx(c,n_c,x,n_x,y,n_y,z-1,n_z)];

		//y down
		if( y > 0 )
			grad_val += 2.0f*dy[idx(c,n_c,x,n_x,y-1,n_y,z,n_z)]*ry[idx(c,n_c,x,n_x,y-1,n_y,z,n_z)];

		//x down
		if( x > 0 )
			grad_val += 2.0f*dy[idx(c,n_c,x-1,n_x,y,n_y,z,n_z)]*rx[idx(c,n_c,x-1,n_x,y,n_y,z,n_z)];

		//z up
		if( z < n_z - 1)
			grad_val += 2.0f*dy[idx(c,n_c,x,n_x,y,n_y,z+1,n_z)]*rz[idx(c,n_c,x,n_x,y,n_y,z,n_z)];

		//y up
		if ( y < n_y - 1)
			grad_val += 2.0f*dy[idx(c,n_c,x,n_x,y+1,n_y,z,n_z)]*ry[idx(c,n_c,x,n_x,y,n_y,z,n_z)];

		//x up
		if ( x < n_x - 1)
			grad_val += 2.0f*dy[idx(c,n_c,x+1,n_x,y,n_y,z,n_z)]*rx[idx(c,n_c,x,n_x,y,n_y,z,n_z)];

		du[idx(c,n_c,x,n_x,y,n_y,z,n_z)] = tau*grad_val + (1.0f-tau)*du[idx(c,n_c,x,n_x,y,n_y,z,n_z)];
	}
}

void get_gradient_for_u_channels_first(const float* dy, const float* rx, const float* ry, float* du, const int n_x, const int n_y, const int n_c, const float tau){
	for(int c = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++){
		float grad_val = 0.0f;
		//y down
		if( y > 0 )
			grad_val += 2.0f*dy[idx(c,n_c,x,n_x,y-1,n_y)]*ry[idx(c,n_c,x,n_x,y-1,n_y)];

		//x down
		if( x > 0 )
			grad_val += 2.0f*dy[idx(c,n_c,x-1,n_x,y,n_y)]*rx[idx(c,n_c,x-1,n_x,y,n_y)];
		
		//y up
		if ( y < n_y - 1)
			grad_val += 2.0f*dy[idx(c,n_c,x,n_x,y+1,n_y)]*ry[idx(c,n_c,x,n_x,y,n_y)];

		//x up
		if ( x < n_x - 1)
			grad_val += 2.0f*dy[idx(c,n_c,x+1,n_x,y,n_y)]*rx[idx(c,n_c,x,n_x,y,n_y)];

		du[idx(c,n_c,x,n_x,y,n_y)] = tau*grad_val + (1.0f-tau)*du[idx(c,n_c,x,n_x,y,n_y)];
	}
}

void get_gradient_for_u_channels_first(const float* dy, const float* rx, float* du, const int n_x, const int n_c, const float tau){
	for(int c = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++){
		float grad_val = 0.0f;
		//x down
		if( x > 0 )
			grad_val += 2.0f*dy[idx(c,n_c,x-1,n_x)]*rx[idx(c,n_c,x-1,n_x)];
		
		//x up
		if ( x < n_x - 1)
			grad_val += 2.0f*dy[idx(c,n_c,x+1,n_x)]*rx[idx(c,n_c,x,n_x)];

		du[idx(c,n_c,x,n_x)] = tau*grad_val + (1.0f-tau)*du[idx(c,n_c,x,n_x)];
	}
}


void get_gradient_for_u(const CPU_DEVICE & dev, const float* dy, const float *const *const r, float* const du, const int dim, const int* const n, const int n_c, const float tau){
    if(dev.channels_first)
        switch(dim){
            case 1:
                get_gradient_for_u_channels_first(dy, r[0], du, n[0], n_c, tau);
                break;
            case 2:
                get_gradient_for_u_channels_first(dy, r[0], r[1], du, n[0], n[1], n_c, tau);
                break;
            case 3:
                get_gradient_for_u_channels_first(dy, r[0], r[1], r[2], du, n[0], n[1], n[2], n_c, tau);
                break;
        }
    else
        switch(dim){
            case 1:
                get_gradient_for_u_channels_last(dy, r[0], du, n[0], n_c, tau);
                break;
            case 2:
                get_gradient_for_u_channels_last(dy, r[0], r[1], du, n[0], n[1], n_c, tau);
                break;
            case 3:
                get_gradient_for_u_channels_last(dy, r[0], r[1], r[2], du, n[0], n[1], n[2], n_c, tau);
                break;
        }
        
}

void get_reg_gradients_channels_last(const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c, const float tau){
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int z = 0; z < n_z; z++)
	for(int c = 0; c < n_c; c++){
		
		//for z
		if( z < n_z - 1 ){
			float up_contra = (2.0f*u[idxc(x,n_x,y,n_y,z+1,n_z,c,n_c)]-1.0f) * g[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];
			float dn_contra = (2.0f*u[idxc(x,n_x,y,n_y,z,n_z,c,n_c)]-1.0f) * g[idxc(x,n_x,y,n_y,z+1,n_z,c,n_c)];
			g_rz[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += tau * (up_contra + dn_contra);
		}

		//for y
		if( y < n_y - 1 ){
			float up_contra = (2.0f*u[idxc(x,n_x,y+1,n_y,z,n_z,c,n_c)]-1.0f) * g[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];
			float dn_contra = (2.0f*u[idxc(x,n_x,y,n_y,z,n_z,c,n_c)]-1.0f) * g[idxc(x,n_x,y+1,n_y,z,n_z,c,n_c)];
			g_ry[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += tau * (up_contra + dn_contra);
		}

		//for x
		if( x < n_x - 1){
			float up_contra = (2.0f*u[idxc(x+1,n_x,y,n_y,z,n_z,c,n_c)]-1.0f) * g[idxc(x,n_x,y,n_y,z,n_z,c,n_c)];
			float dn_contra = (2.0f*u[idxc(x,n_x,y,n_y,z,n_z,c,n_c)]-1.0f) * g[idxc(x+1,n_x,y,n_y,z,n_z,c,n_c)];
			g_rx[idxc(x,n_x,y,n_y,z,n_z,c,n_c)] += tau * (up_contra + dn_contra);
		}
	}

}

void get_reg_gradients_channels_last(const float* g, const float* u, float* g_rx, float* g_ry, const int n_x, const int n_y, const int n_c, const float tau){
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int c = 0; c < n_c; c++){

		//for y
		if( y < n_y - 1 ){
			float up_contra = (2.0f*u[idxc(x,n_x,y+1,n_y,c,n_c)]-1.0f) * g[idxc(x,n_x,y,n_y,c,n_c)];
			float dn_contra = (2.0f*u[idxc(x,n_x,y,n_y,c,n_c)]-1.0f) * g[idxc(x,n_x,y+1,n_y,c,n_c)];
			g_ry[idxc(x,n_x,y,n_y,c,n_c)] += tau * (up_contra + dn_contra);
		}

		//for x
		if( x < n_x - 1){
			float up_contra = (2.0f*u[idxc(x+1,n_x,y,n_y,c,n_c)]-1.0f) * g[idxc(x,n_x,y,n_y,c,n_c)];
			float dn_contra = (2.0f*u[idxc(x,n_x,y,n_y,c,n_c)]-1.0f) * g[idxc(x+1,n_x,y,n_y,c,n_c)];
			g_rx[idxc(x,n_x,y,n_y,c,n_c)] += tau * (up_contra + dn_contra);
		}
	}

}

void get_reg_gradients_channels_last(const float* g, const float* u, float* g_rx, const int n_x, const int n_c, const float tau){
	for(int x = 0; x < n_x; x++)
	for(int c = 0; c < n_c; c++){

		//for x
		if( x < n_x - 1){
			float up_contra = (2.0f*u[idxc(x+1,n_x,c,n_c)]-1.0f) * g[idxc(x,n_x,c,n_c)];
			float dn_contra = (2.0f*u[idxc(x,n_x,c,n_c)]-1.0f) * g[idxc(x+1,n_x,c,n_c)];
			g_rx[idxc(x,n_x,c,n_c)] += tau * (up_contra + dn_contra);
		}
	}
}

void get_reg_gradients_channels_first(const float* g, const float* u, float* g_rx, float* g_ry, float* g_rz, const int n_x, const int n_y, const int n_z, const int n_c, const float tau){
	for(int c = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++)
	for(int z = 0; z < n_z; z++){
		
		//for z
		if( z < n_z - 1 ){
			float up_contra = (2.0f*u[idx(c,n_c,x,n_x,y,n_y,z+1,n_z)]-1.0f) * g[idx(c,n_c,x,n_x,y,n_y,z,n_z)];
			float dn_contra = (2.0f*u[idx(c,n_c,x,n_x,y,n_y,z,n_z)]-1.0f) * g[idx(c,n_c,x,n_x,y,n_y,z+1,n_z)];
			g_rz[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += tau * (up_contra + dn_contra);
		}

		//for y
		if( y < n_y - 1 ){
			float up_contra = (2.0f*u[idx(c,n_c,x,n_x,y+1,n_y,z,n_z)]-1.0f) * g[idx(c,n_c,x,n_x,y,n_y,z,n_z)];
			float dn_contra = (2.0f*u[idx(c,n_c,x,n_x,y,n_y,z,n_z)]-1.0f) * g[idx(c,n_c,x,n_x,y+1,n_y,z,n_z)];
			g_ry[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += tau * (up_contra + dn_contra);
		}

		//for x
		if( x < n_x - 1){
			float up_contra = (2.0f*u[idx(c,n_c,x+1,n_x,y,n_y,z,n_z)]-1.0f) * g[idx(c,n_c,x,n_x,y,n_y,z,n_z)];
			float dn_contra = (2.0f*u[idx(c,n_c,x,n_x,y,n_y,z,n_z)]-1.0f) * g[idx(c,n_c,x+1,n_x,y,n_y,z,n_z)];
			g_rx[idx(c,n_c,x,n_x,y,n_y,z,n_z)] += tau * (up_contra + dn_contra);
		}
	}
}

void get_reg_gradients_channels_first(const float* g, const float* u, float* g_rx, float* g_ry, const int n_x, const int n_y, const int n_c, const float tau){
	for(int c = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++)
	for(int y = 0; y < n_y; y++){

		//for y
		if( y < n_y - 1 ){
			float up_contra = (2.0f*u[idx(c,n_c,x,n_x,y+1,n_y)]-1.0f) * g[idx(c,n_c,x,n_x,y,n_y)];
			float dn_contra = (2.0f*u[idx(c,n_c,x,n_x,y,n_y)]-1.0f) * g[idx(c,n_c,x,n_x,y+1,n_y)];
			g_ry[idx(c,n_c,x,n_x,y,n_y)] += tau * (up_contra + dn_contra);
		}

		//for x
		if( x < n_x - 1){
			float up_contra = (2.0f*u[idx(c,n_c,x+1,n_x,y,n_y)]-1.0f) * g[idx(c,n_c,x,n_x,y,n_y)];
			float dn_contra = (2.0f*u[idx(c,n_c,x,n_x,y,n_y)]-1.0f) * g[idx(c,n_c,x+1,n_x,y,n_y)];
			g_rx[idx(c,n_c,x,n_x,y,n_y)] += tau * (up_contra + dn_contra);
		}
	}
}

void get_reg_gradients_channels_first(const float* g, const float* u, float* g_rx, const int n_x, const int n_c, const float tau){
	for(int c = 0; c < n_c; c++)
	for(int x = 0; x < n_x; x++){

		//for x
		if( x < n_x - 1){
			float up_contra = (2.0f*u[idx(c,n_c,x+1,n_x)]-1.0f) * g[idx(c,n_c,x,n_x)];
			float dn_contra = (2.0f*u[idx(c,n_c,x,n_x)]-1.0f) * g[idx(c,n_c,x+1,n_x)];
			g_rx[idx(c,n_c,x,n_x)] += tau * (up_contra + dn_contra);
		}
	}
}

void populate_reg_mean_gradients_and_add(const CPU_DEVICE& dev, const float* g, const float* u, float *const *const g_r, const int dim, const int* const n, const int n_c, const float tau){
    if(dev.channels_first)
        switch(dim){
            case 1:
                get_reg_gradients_channels_first(g, u, g_r[0], n[0], n_c, tau);
                break;
            case 2:
                get_reg_gradients_channels_first(g, u, g_r[0], g_r[1], n[0], n[1], n_c, tau);
                break;
            case 3:
                get_reg_gradients_channels_first(g, u, g_r[0], g_r[1], g_r[2], n[0], n[1], n[2], n_c, tau);
                break;
        }
    else
        switch(dim){
            case 1:
                get_reg_gradients_channels_last(g, u, g_r[0], n[0], n_c, tau);
                break;
            case 2:
                get_reg_gradients_channels_last(g, u, g_r[0], g_r[1], n[0], n[1], n_c, tau);
                break;
            case 3:
                get_reg_gradients_channels_last(g, u, g_r[0], g_r[1], g_r[2], n[0], n[1], n[2], n_c, tau);
                break;
        }
}
	

void compute_source_flow( const float* u, float* ps, const float* pt, const float* div, const float icc, const int n_c, const int n_s){
    for(int s = 0; s < n_s; s++){
        ps[s] = icc;
        for(int c = 0; c < n_c; c++)
            ps[s] += pt[idxc(s,n_s,c,n_c)] + div[idxc(s,n_s,c,n_c)] - u[idxc(s,n_s,c,n_c)] * icc;
        ps[s] /= n_c;
    }
}

void compute_sink_flow( const float* u, const float* ps, float* pt, const float* div, const float* d, const float icc, const int n_c, const int n_s){
    for(int s = 0; s < n_s; s++)
        for (int c = 0; c < n_c; c++) {
            int i = idxc(s,n_s,c,n_c);
            pt[i] = ps[s] - div[i] + u[i] * icc;
            if( pt[i] > -d[i] )
                pt[i] = -d[i];
        }
}
    
void compute_multipliers( float* erru, float* u, const float* ps, const float* pt, const float* div, const float cc, const int n_c, const int n_s){
    for(int s = 0; s < n_s; s++)
        for (int c = 0; c < n_c; c++) {
            int i = idxc(s,n_s,c,n_c);
            erru[i] = cc * (ps[s] - div[i] - pt[i]);
            u[i] += erru[i];
        }
}

void update_source_sink_multiplier_potts_channels_last( float* erru, float* u, float* ps, float* pt, const float* div, const float* d, const float cc, const float icc, const int n_c, const int n_s){

    for(int s = 0; s < n_s; s++){
        ps[s] = icc;
        for(int c = 0; c < n_c; c++)
            ps[s] += pt[idxc(s,n_s,c,n_c)] + div[idxc(s,n_s,c,n_c)] - u[idxc(s,n_s,c,n_c)] * icc;
        ps[s] /= n_c;
        
        for (int c = 0; c < n_c; c++) {
            int i = idxc(s,n_s,c,n_c);
            pt[i] = ps[s] - div[i] + u[i] * icc;
            if( pt[i] > -d[i] )
                pt[i] = -d[i];
            
            erru[i] = cc * (ps[s] - div[i] - pt[i]);
            u[i] += erru[i];
            if(erru[i] < 0.0f)
                erru[i] = -erru[i];
        }
    }
}

void update_source_sink_multiplier_potts_channels_first( float* erru, float* u, float* ps, float* pt, const float* div, const float* d, const float cc, const float icc, const int n_c, const int n_s){

    for(int s = 0; s < n_s; s++){
        ps[s] = icc;
        for(int c = 0; c < n_c; c++){
            ps[s] += pt[idx(c,n_c,s,n_s)];
            ps[s] += div[idx(c,n_c,s,n_s)];
            ps[s] -= u[idx(c,n_c,s,n_s)] * icc;
        }
        ps[s] /= n_c;
        
        for (int c = 0; c < n_c; c++) {
            int i = idx(c,n_c,s,n_s);
            pt[i] = ps[s] - div[i] + u[i] * icc;
            if( pt[i] > -d[i] )
                pt[i] = -d[i];
            
            erru[i] = cc * (ps[s] - div[i] - pt[i]);
            u[i] += erru[i];
            if(erru[i] < 0.0f)
                erru[i] = -erru[i];
        }
    }
}

void update_source_sink_multiplier_potts(const CPU_DEVICE& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_c, const int n_s){
    if(dev.channels_first)
        update_source_sink_multiplier_potts_channels_first(erru, u, ps, pt, div, d, cc, icc, n_c, n_s);
    else
        update_source_sink_multiplier_potts_channels_last(erru, u, ps, pt, div, d, cc, icc, n_c, n_s);
}

void update_source_sink_multiplier_binary(const CPU_DEVICE& dev, float* ps, float* pt, const float* div, float* u, float* erru, const float* d, const float cc, const float icc, const int n_s){
    for(int s = 0; s < n_s; s++){
        float d_tmp = d[s];
        float pt_tmp = pt[s];
        float div_tmp = div[s];
        float u_tmp = u[s];
        
        float ps_tmp = icc + pt_tmp + div_tmp - u_tmp * icc;
		if( d_tmp < 0.0f )
			ps_tmp = 0.0f;
		else if( ps_tmp > d_tmp)
			ps_tmp = d_tmp;
        ps[s] = ps_tmp;
		
        pt_tmp = ps_tmp - div_tmp + u_tmp * icc;
		if( d_tmp > 0.0f )
			pt_tmp = 0.0f;
		else if( pt_tmp > -d_tmp)
			pt_tmp = -d_tmp;
        pt[s] = pt_tmp;
        
		float erru_tmp = cc * (ps_tmp - div_tmp - pt_tmp);
		u[s] += erru_tmp;
        erru[s] = (erru_tmp > 0.0f) ? erru_tmp : -erru_tmp;
    }
}

void calc_capacity_potts_channels_last(float* g, const float* u, const float* ps, const float* pt, const float* div, const int n_s, const int n_c, const float tau, const float icc){
    for(int s = 0, cs = 0; s < n_s; s++)
    for(int c = 0; c < n_c; c++, cs++)
        g[cs] = tau * (div[cs] + pt[cs] - ps[s] - u[cs] * icc);
}

void calc_capacity_potts_channels_first(float* g, const float* u, const float* ps, const float* pt, const float* div, const int n_s, const int n_c, const float tau, const float icc){
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int s = 0; s < n_s; s++, cs++)
        g[cs] = tau*(div[cs] + pt[cs] - ps[s] - u[cs] * icc);
}

void calc_capacity_potts(const CPU_DEVICE & dev, float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const int n_c, const float icc, const float tau){
    if(dev.channels_first)
        calc_capacity_potts_channels_first(g,u,ps,pt,div,n_s,n_c,tau,icc);
    else
        calc_capacity_potts_channels_last(g,u,ps,pt,div,n_s,n_c,tau,icc);
}

void calc_capacity_binary(const CPU_DEVICE & dev, float* g, const float* div, const float* ps, const float* pt, const float* u, const int n_s, const float icc, const float tau){
    for(int s = 0; s < n_s; s++)
		g[s] = tau * (div[s] + pt[s] - ps[s] - u[s] * icc);
}

void update_spatial_flows__update_specific_flow(const int cs, const int csn, const float* g, float* p, const float* r){
    p[cs] += g[cs] - g[csn];
    if (p[cs] > r[cs])
        p[cs] = r[cs];
    else if (p[cs] < -r[cs])
        p[cs] = -r[cs];
}

void update_spatial_flows__update_specific_star_flow(const int cs, const int csn, const float* g, float* p, const float* r, const float* l){
    p[cs] += g[cs] - g[csn];
    if( p[cs]*l[cs] < 0.0f){
        if (p[cs] > r[cs])
            p[cs] = r[cs];
        else if (p[cs] < -r[cs])
            p[cs] = -r[cs];
    }
}

void update_spatial_flows_channels_last(const float* g, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float * rz, const int n_c, const int n_x, const int n_y, const int n_z){
	
    for(int x = 0, s = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++, s++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,z,n_z,c,n_c);
            if (x < n_x-1) {
                int sxm = idxc(x+1,n_x,y,n_y,z,n_z,c,n_c);
                update_spatial_flows__update_specific_flow(cs,sxm,g,px,rx);
            }
            if (y < n_y-1){
                int sym = idxc(x,n_x,y+1,n_y,z,n_z,c,n_c);
                update_spatial_flows__update_specific_flow(cs,sym,g,py,ry);
            }
            if (z < n_z-1){
                int szm = idxc(x,n_x,y,n_y,z+1,n_z,c,n_c);
                update_spatial_flows__update_specific_flow(cs,szm,g,pz,rz);
            }
        }
    }
    
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,z,n_z,c,n_c);
            div[cs] = -px[cs]-py[cs]-pz[cs];
            if (x > 0) {
                int sxm = idxc(x-1,n_x,y,n_y,z,n_z,c,n_c);
                div[cs] += px[sxm];
            }
            if (y > 0){
                int sym = idxc(x,n_x,y-1,n_y,z,n_z,c,n_c);
                div[cs] += py[sym];
            }
            if (z > 0){
                int szm = idxc(x,n_x,y,n_y,z-1,n_z,c,n_c);
                div[cs] += pz[szm];
            }
        }
    }
            
}

void update_spatial_star_flows_channels_last(const float* g, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float * rz, const float* lx, const float* ly, const float * lz, const int n_c, const int n_x, const int n_y, const int n_z){
	
    for(int x = 0, s = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++, s++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,z,n_z,c,n_c);
            if (x < n_x-1) {
                int sxm = idxc(x+1,n_x,y,n_y,z,n_z,c,n_c);
                update_spatial_flows__update_specific_star_flow(cs,sxm,g,px,rx,lx);
            }
            if (y < n_y-1){
                int sym = idxc(x,n_x,y+1,n_y,z,n_z,c,n_c);
                update_spatial_flows__update_specific_star_flow(cs,sym,g,py,ry,ly);
            }
            if (z < n_z-1){
                int szm = idxc(x,n_x,y,n_y,z+1,n_z,c,n_c);
                update_spatial_flows__update_specific_star_flow(cs,szm,g,pz,rz,lz);
            }
        }
    }
    
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,z,n_z,c,n_c);
            div[cs] = -px[cs]-py[cs]-pz[cs];
            if (x > 0) {
                int sxm = idxc(x-1,n_x,y,n_y,z,n_z,c,n_c);
                div[cs] += px[sxm];
            }
            if (y > 0){
                int sym = idxc(x,n_x,y-1,n_y,z,n_z,c,n_c);
                div[cs] += py[sym];
            }
            if (z > 0){
                int szm = idxc(x,n_x,y,n_y,z-1,n_z,c,n_c);
                div[cs] += pz[szm];
            }
        }
    }
            
}

void update_spatial_flows_channels_last(const float* g, float* div, float* px, float* py, const float* rx, const float* ry, const int n_c, const int n_x, const int n_y){
	
    for(int x = 0, s = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++,s++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,c,n_c);
            if (x < n_x-1) {
                int sxm = idxc(x+1,n_x,y,n_y,c,n_c);
                update_spatial_flows__update_specific_flow(cs,sxm,g,px,rx);
            }
            if (y < n_y-1){
                int sym = idxc(x,n_x,y+1,n_y,c,n_c);
                update_spatial_flows__update_specific_flow(cs,sym,g,py,ry);
            }
        }
    }
    
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,c,n_c);
            div[cs] = -px[cs]-py[cs];
            if (x > 0) {
                int sxm = idxc(x-1,n_x,y,n_y,c,n_c);
                div[cs] += px[sxm];
            }
            if (y > 0){
                int sym = idxc(x,n_x,y-1,n_y,c,n_c);
                div[cs] += py[sym];
            }
        }
    }
            
}

void update_spatial_star_flows_channels_last(const float* g, float* div, float* px, float* py, const float* rx, const float* ry, const float* lx, const float* ly, const int n_c, const int n_x, const int n_y){
	
    for(int x = 0, s = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++,s++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,c,n_c);
            if (x < n_x-1) {
                int sxm = idxc(x+1,n_x,y,n_y,c,n_c);
                update_spatial_flows__update_specific_star_flow(cs,sxm,g,px,rx,lx);
            }
            if (y < n_y-1){
                int sym = idxc(x,n_x,y+1,n_y,c,n_c);
                update_spatial_flows__update_specific_star_flow(cs,sym,g,py,ry,ly);
            }
        }
    }
    
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,y,n_y,c,n_c);
            div[cs] = -px[cs]-py[cs];
            if (x > 0) {
                int sxm = idxc(x-1,n_x,y,n_y,c,n_c);
                div[cs] += px[sxm];
            }
            if (y > 0){
                int sym = idxc(x,n_x,y-1,n_y,c,n_c);
                div[cs] += py[sym];
            }
        }
    }
            
}

void update_spatial_flows_channels_last(const float* g, float* div, float* px, const float* rx, const int n_c, const int n_x){
    
    for(int x = 0, s = 0; x < n_x; x++, s++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,c,n_c);
            if (x != n_x-1) {
                int sxm = idxc(x+1,n_x,c,n_c);
                update_spatial_flows__update_specific_flow(cs,sxm,g,px,rx);
            }
        }
    }
    
    for(int x = 0; x < n_x; x++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,c,n_c);
            div[cs] = -px[cs];
            if (x > 0) {
                int sxm = idxc(x-1,n_x,c,n_c);
                div[cs] += px[sxm];
            }
        }
    }
            
}
void update_spatial_star_flows_channels_last(const float* g, float* div, float* px, const float* rx, const float* lx, const int n_c, const int n_x){
    
    for(int x = 0, s = 0; x < n_x; x++, s++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,c,n_c);
            if (x != n_x-1) {
                int sxm = idxc(x+1,n_x,c,n_c);
                update_spatial_flows__update_specific_star_flow(cs,sxm,g,px,rx,lx);
            }
        }
    }
    
    for(int x = 0; x < n_x; x++){
        for(int c = 0; c < n_c; c++){
            int cs = idxc(x,n_x,c,n_c);
            div[cs] = -px[cs];
            if (x > 0) {
                int sxm = idxc(x-1,n_x,c,n_c);
                div[cs] += px[sxm];
            }
        }
    }
            
}

void update_spatial_flows_channels_first(const float* g, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float * rz, const int n_c, const int n_x, const int n_y, const int n_z){

    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++, cs++){
        if (x < n_x-1) {
            int sxm = cs + n_y*n_z;
            update_spatial_flows__update_specific_flow(cs,sxm,g,px,rx);
        }
        if (y < n_y-1){
            int sym = cs + n_z;
            update_spatial_flows__update_specific_flow(cs,sym,g,py,ry);
        }
        if (z < n_z-1){
            int szm = cs + 1;
            update_spatial_flows__update_specific_flow(cs,szm,g,pz,rz);
        }
    }
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++,cs++){
        div[cs] = -px[cs]-py[cs]-pz[cs];
        if (x > 0) {
            int sxm = cs - n_y*n_z;
            div[cs] += px[sxm];
        }
        if (y > 0){
            int sym = cs - n_z;;
            div[cs] += py[sym];
        }
        if (z > 0){
            int szm = cs - 1;
            div[cs] += pz[szm];
        }
    }
            
}

void update_spatial_star_flows_channels_first(const float* g, float* div, float* px, float* py, float* pz, const float* rx, const float* ry, const float * rz, const float* lx, const float* ly, const float * lz, const int n_c, const int n_x, const int n_y, const int n_z){

    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++, cs++){
        if (x < n_x-1) {
            int sxm = cs + n_y*n_z;
            update_spatial_flows__update_specific_star_flow(cs,sxm,g,px,rx,lx);
        }
        if (y < n_y-1){
            int sym = cs + n_z;
            update_spatial_flows__update_specific_star_flow(cs,sym,g,py,ry,ly);
        }
        if (z < n_z-1){
            int szm = cs + 1;
            update_spatial_flows__update_specific_star_flow(cs,szm,g,pz,rz,lz);
        }
    }
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++)
    for(int z = 0; z < n_z; z++,cs++){
        div[cs] = -px[cs]-py[cs]-pz[cs];
        if (x > 0) {
            int sxm = cs - n_y*n_z;
            div[cs] += px[sxm];
        }
        if (y > 0){
            int sym = cs - n_z;;
            div[cs] += py[sym];
        }
        if (z > 0){
            int szm = cs - 1;
            div[cs] += pz[szm];
        }
    }
            
}

void update_spatial_flows_channels_first(const float* g, float* div, float* px, float* py, const float* rx, const float* ry, const int n_c, const int n_x, const int n_y){

    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++,cs++){
        if (x < n_x-1) {
            int sxm = cs + n_y;
            update_spatial_flows__update_specific_flow(cs,sxm,g,px,rx);
        }
        if (y < n_y-1){
            int sym = cs + 1;
            update_spatial_flows__update_specific_flow(cs,sym,g,py,ry);
        }
    }
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++,cs++){
        div[cs] = -px[cs]-py[cs];
        if (x > 0) {
            int sxm = cs - n_y;
            div[cs] += px[sxm];
        }
        if (y > 0){
            int sym = cs - 1;
            div[cs] += py[sym];
        }
    }
            
}

void update_spatial_star_flows_channels_first(const float* g, float* div, float* px, float* py, const float* rx, const float* ry, const float* lx, const float* ly, const int n_c, const int n_x, const int n_y){

    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++,cs++){
        if (x < n_x-1) {
            int sxm = cs + n_y;
            update_spatial_flows__update_specific_star_flow(cs,sxm,g,px,rx,lx);
        }
        if (y < n_y-1){
            int sym = cs + 1;
            update_spatial_flows__update_specific_star_flow(cs,sym,g,py,ry,ly);
        }
    }
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++)
    for(int y = 0; y < n_y; y++,cs++){
        div[cs] = -px[cs]-py[cs];
        if (x > 0) {
            int sxm = cs - n_y;
            div[cs] += px[sxm];
        }
        if (y > 0){
            int sym = cs - 1;
            div[cs] += py[sym];
        }
    }
            
}


void update_spatial_flows_channels_first(const float* g, float* div, float* px, const float* rx, const int n_c, const int n_x){

    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++,cs++){
        if (x < n_x-1) {
            int sxm = cs + 1;
            update_spatial_flows__update_specific_flow(cs,sxm,g,px,rx);
        }
    }
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++, cs++){
        div[cs] = -px[cs];
        if (x > 0) {
            int sxm = cs - 1;
            div[cs] += px[sxm];
        }
    }

}

void update_spatial_star_flows_channels_first(const float* g, float* div, float* px, const float* rx, const float* lx, const int n_c, const int n_x){

    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++,cs++){
        if (x < n_x-1) {
            int sxm = cs + 1;
            update_spatial_flows__update_specific_star_flow(cs,sxm,g,px,rx,lx);
        }
    }
    
    for(int c = 0, cs = 0; c < n_c; c++)
    for(int x = 0; x < n_x; x++, cs++){
        div[cs] = -px[cs];
        if (x > 0) {
            int sxm = cs - 1;
            div[cs] += px[sxm];
        }
    }

}
    
void update_spatial_flows(const CPU_DEVICE& dev, const float* const g, float* const div, float *const *const p, const float *const *const r, const int dim, const int* const n, const int n_c){
    if(dev.channels_first)
        switch(dim){
            case 1:
                update_spatial_flows_channels_first(g, div, p[0], r[0], n_c, n[0]);
                break;
            case 2:
                update_spatial_flows_channels_first(g, div, p[0], p[1], r[0], r[1], n_c, n[0], n[1]);
                break;
            case 3:
                update_spatial_flows_channels_first(g, div, p[0], p[1], p[2], r[0], r[1], r[2], n_c, n[0], n[1], n[2]);
                break;
        }
    else
        switch(dim){
            case 1:
                update_spatial_flows_channels_last(g, div, p[0], r[0], n_c, n[0]);
                break;
            case 2:
                update_spatial_flows_channels_last(g, div, p[0], p[1], r[0], r[1], n_c, n[0], n[1]);
                break;
            case 3:
                update_spatial_flows_channels_last(g, div, p[0], p[1], p[2], r[0], r[1], r[2], n_c, n[0], n[1], n[2]);
                break;
        }
}
    
void update_spatial_star_flows(const CPU_DEVICE& dev, const float* const g, float* const div, float *const *const p, const float *const *const r, const float *const *const l, const int dim, const int* const n, const int n_c){
    if(dev.channels_first)
        switch(dim){
            case 1:
                update_spatial_star_flows_channels_first(g, div, p[0], r[0], l[0], n_c, n[0]);
                break;
            case 2:
                update_spatial_star_flows_channels_first(g, div, p[0], p[1], r[0], r[1], l[0], l[1], n_c, n[0], n[1]);
                break;
            case 3:
                update_spatial_star_flows_channels_first(g, div, p[0], p[1], p[2], r[0], r[1], r[2], l[0], l[1], l[2], n_c, n[0], n[1], n[2]);
                break;
        }
    else
        switch(dim){
            case 1:
                update_spatial_star_flows_channels_last(g, div, p[0], r[0], l[0], n_c, n[0]);
                break;
            case 2:
                update_spatial_star_flows_channels_last(g, div, p[0], p[1], r[0], r[1], l[0], l[1], n_c, n[0], n[1]);
                break;
            case 3:
                update_spatial_star_flows_channels_last(g, div, p[0], p[1], p[2], r[0], r[1], r[2], l[0], l[1], l[2], n_c, n[0], n[1], n[2]);
                break;
        }
}

void init_flows_binary(const CPU_DEVICE & dev, const float* d, float* ps, float* pt, float* u, const int n_s){
    for(int s = 0; s < n_s; s++){
		float d_val = d[s];
		if (d_val > 0.0f){
			ps[s] = d_val;
            pt[s] = 0.0f;
            u[s] = 1.0;
        }else{
			pt[s] = -d_val;
            ps[s] = 0.0f;
            u[s] = 0.0;
        }
	}
}

void init_flows_potts_channels_last(const float* d, float* ps, float* pt, float* u, const int n_c, const int n_s){
    for(int s = 0; s < n_s; s++){
        float max_d = -std::numeric_limits<float>::infinity();
        for(int c = 0; c < n_c; c++){
            int cs = idxc(s,n_s,c,n_c);
            if( max_d < d[cs] )
                max_d = d[cs];
        }
        //cs -= n_c;
        ps[s] = -max_d;
        for(int c = 0; c < n_c; c++){
            int cs = idxc(s,n_s,c,n_c);
            pt[cs] = -max_d;
            if (d[cs] == max_d)
                u[cs] = 1.0f;
            else
                u[cs] = 0.0f;
        }
    }    
}

void init_flows_potts_channels_first(const float* d, float* ps, float* pt, float* u, const int n_c, const int n_s){
    for(int s = 0; s < n_s; s++){
        float max_d = -std::numeric_limits<float>::infinity();
        for(int c = 0; c < n_c; c++){
            int cs = idx(c,n_c,s,n_s);
            if( max_d < d[cs] )
                max_d = d[cs];
        }
        //cs -= n_c;
        ps[s] = -max_d;
        for(int c = 0; c < n_c; c++){
            int cs = idx(c,n_c,s,n_s);
            pt[cs] = -max_d;
            if (d[cs] == max_d)
                u[cs] = 1.0f;
            else
                u[cs] = 0.0f;
        }
    }     
}

void init_flows_potts(const CPU_DEVICE & dev,const float* d, float* ps, float* pt, float* u, const int n_s, const int n_c){
    if(dev.channels_first)
        init_flows_potts_channels_first(d, ps, pt, u, n_c, n_s);
    else
        init_flows_potts_channels_last(d, ps, pt, u, n_c, n_s);
}

void init_flows_channels_last(const float* d, float* ps, const int n_c, const int n_s){
    for(int s = 0; s < n_s; s++){
        float max_d = -std::numeric_limits<float>::infinity();
        for(int c = 0; c < n_c; c++){
            int cs = idxc(s,n_s,c,n_c);
            if( max_d < d[cs] )
                max_d = d[cs];
        }
        //cs -= n_c;
        ps[s] = -max_d;
    }
            
}
void init_flows_channels_first(const float* d, float* ps, const int n_c, const int n_s){
    for(int s = 0; s < n_s; s++)
        ps[s] = std::numeric_limits<float>::infinity();
    for(int c = 0, cs = 0; c < n_c; c++)
        for(int s = 0; s < n_s; s++, cs++)
            if( ps[s] > d[cs] )
                ps[s] = d[cs];
}

void init_flows(const CPU_DEVICE & dev, const float* d, float* ps, const int n_c, const int n_s){
    if(dev.channels_first)
        init_flows_channels_first(d, ps, n_c, n_s);
    else
        init_flows_channels_last(d, ps, n_c, n_s);
}