#ifndef KERNEL_SIGMA_H
#define KERNEL_SIGMA_H


const char* computeSigmaKernelSource = R"(
__kernel void compute_sigma(__global const float* wtNodeProd, __global float* sigma, int in_lyr_sz, int out_lyr_sz) {
    int j = get_global_id(0);

    if (j < out_lyr_sz) {
        sigma[j] = 0.0f;
        for (int i = 0; i < in_lyr_sz; i++) {
            sigma[j] += wtNodeProd[(i * out_lyr_sz) + j];
        }
    }
}
)";
#endif