#ifndef KERNEL_WT_NODE_PROD_H
#define KERNEL_WT_NODE_PROD_H


const char* computeWtNodeProdKernelSource = R"(
__kernel void compute_wt_node_prod(__global const float* wtMtx, __global const float* in_node_vals, __global float* wtNodeProd, int in_lyr_sz, int out_lyr_sz) {
    int j = get_global_id(0);
    int i = get_global_id(1);

    if (j < out_lyr_sz && i < in_lyr_sz) {
        wtNodeProd[(i * out_lyr_sz) + j] = ((wtMtx[(i * out_lyr_sz) + j]) * in_node_vals[i]);
    }
}
)";
#endif