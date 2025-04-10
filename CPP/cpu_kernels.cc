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

