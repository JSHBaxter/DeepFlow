#include "common.h"

int product(const int d, const int* const di){
    int r = 1;
    for(int i = 0; i < d; i++)
        r *= di[i];
    return r;
}