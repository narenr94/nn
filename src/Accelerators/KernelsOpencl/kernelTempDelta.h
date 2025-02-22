#ifndef KERNEL_TEMPDELTA_H
#define KERNEL_TEMPDELTA_H


const char* computeTempDeltaKernelSource = R"(
__kernel void compute_temp_delta(__global const float* wtNodeProd, __global float* temp, int in_lyr_sz, int out_lyr_sz) {
    int j = get_global_id(0);

    if (j < in_lyr_sz) {
        temp[j] = 0.0f;
        for (int k = 0; k < out_lyr_sz; k++) {
            temp[j] += wtNodeProd[(j * out_lyr_sz) + k];
        }
    }
}
)";
#endif