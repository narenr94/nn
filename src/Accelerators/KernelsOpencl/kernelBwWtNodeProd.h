#ifndef KERNEL_BW_WT_NODE_PROD_H
#define KERNEL_BW_WT_NODE_PROD_H


const char* computeBwWtNodeProdKernelSource = R"(
__kernel void compute_bw_wt_node_prod(__global const float* wtMtx, __global const float* in_node_vals, __global float* wtNodeProd, int in_lyr_sz, int out_lyr_sz) {
    int j = get_global_id(0);
    int k = get_global_id(1);

    if (j < in_lyr_sz && k < out_lyr_sz) {
        wtNodeProd[(j * out_lyr_sz) + k] = ((wtMtx[(j * out_lyr_sz) + k]) * in_node_vals[k]);
    }
}
)";
#endif