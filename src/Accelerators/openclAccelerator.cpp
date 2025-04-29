#include "openclAccelerator.h"

//kernels
#include "KernelsOpencl/kernelSigma.h"
#include "KernelsOpencl/kernelWtNodeProd.h"
#include "KernelsOpencl/kernelTempDelta.h"
#include "KernelsOpencl/kernelBwWtNodeProd.h"

#include "baseLayer.h"

OpenclAccelerator::OpenclAccelerator(BaseLayer* pLayer):BaseAccelerator(pLayer)
{
    m_pLayer = pLayer;
    // Set up OpenCL context, device, and queue
    platform = cl::Platform::getDefault();
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    device = devices.front();
    context = cl::Context(device);
    queue = cl::CommandQueue(context, device);

    // Build the compute_wt_node_prod kernel
    cl::Program computeWtNodeProdProgram(context, computeWtNodeProdKernelSource);
    computeWtNodeProdProgram.build();
    computeWtNodeProdKernel = cl::Kernel(computeWtNodeProdProgram, "compute_wt_node_prod");

    // Build the compute_sigma kernel
    cl::Program computeSigmaProgram(context, computeSigmaKernelSource);
    computeSigmaProgram.build();
    computeSigmaKernel = cl::Kernel(computeSigmaProgram, "compute_sigma");

    //Build the compute_temp_delta
    cl::Program computeTempDeltaProdProgram(context, computeTempDeltaKernelSource);
    computeTempDeltaProdProgram.build();
    computeTempDeltaKernel = cl::Kernel(computeTempDeltaProdProgram, "compute_temp_delta");

    //Build the compute_bw_wt_node_prod
    cl::Program computecomputeBwWtNodeProdProgram(context, computeBwWtNodeProdKernelSource);
    computecomputeBwWtNodeProdProgram.build();
    computeBwWtNodeProdKernel = cl::Kernel(computecomputeBwWtNodeProdProgram, "compute_bw_wt_node_prod");
}

OpenclAccelerator::~OpenclAccelerator()
{
    queue = nullptr;
    context = nullptr;
    computeWtNodeProdKernel = nullptr;
    computeSigmaKernel = nullptr;
}

void OpenclAccelerator::do_forwardpass_dense_layer()
{
    BaseLayer* in_lyr = m_pLayer->GetPreviousLayer();
    BaseLayer* out_lyr = m_pLayer;

    uint in_lyr_sz = in_lyr->get_num_nodes();
    uint out_lyr_sz = out_lyr->get_num_nodes();

    const float * wtMtx = m_pLayer->get_transform_matrix();

    uint i = 0;
    uint j = 0;

    float* sigma = new float [out_lyr_sz];

    float* inNodeVal = new float [in_lyr_sz];

    for(i = 0; i < in_lyr_sz; i++)
    {
        inNodeVal[i] = in_lyr->get_node_value_idx(i);
    }

    cl::Buffer wtMtxBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * in_lyr_sz * out_lyr_sz, (void*)wtMtx);
    cl::Buffer inNodeValsBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * in_lyr_sz, (void*)inNodeVal);
    cl::Buffer wtNodeProdBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * in_lyr_sz * out_lyr_sz);
    cl::Buffer sigmaBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * out_lyr_sz, sigma);

    // Set kernel arguments
    computeWtNodeProdKernel.setArg(0, wtMtxBuffer);
    computeWtNodeProdKernel.setArg(1, inNodeValsBuffer);
    computeWtNodeProdKernel.setArg(2, wtNodeProdBuffer);
    computeWtNodeProdKernel.setArg(3, in_lyr_sz);
    computeWtNodeProdKernel.setArg(4, out_lyr_sz);

    computeSigmaKernel.setArg(0, wtNodeProdBuffer);
    computeSigmaKernel.setArg(1, sigmaBuffer);
    computeSigmaKernel.setArg(2, in_lyr_sz);
    computeSigmaKernel.setArg(3, out_lyr_sz);

    cl::NDRange global(out_lyr_sz, in_lyr_sz);
    queue.enqueueNDRangeKernel(computeWtNodeProdKernel, cl::NullRange, global);

    cl::NDRange globalSigma(out_lyr_sz);
    queue.enqueueNDRangeKernel(computeSigmaKernel, cl::NullRange, globalSigma);
    queue.enqueueReadBuffer(sigmaBuffer, CL_TRUE, 0, sizeof(float) * out_lyr_sz, sigma);


    for(j = 0; j < out_lyr_sz; j++)
    {
        sigma[j] += out_lyr->get_node_bias_idx(j);
        sigma[j] /= in_lyr->get_num_nodes();
        // sigma = m_pActFunc->apply_act_func(sigma);
        out_lyr->set_node_value(sigma[j], j);
    }

    //apply activation function
    out_lyr->apply_act_func_all_nodes();

    delete [] inNodeVal;
    delete [] sigma;

    wtMtxBuffer = nullptr;
    inNodeValsBuffer = nullptr;
    wtNodeProdBuffer = nullptr;
    sigmaBuffer = nullptr;

}

void OpenclAccelerator::do_backwardpass_dense_layer_output_layer(float* pfExpOut, BaseLossFunction* lossFunc)
{
    //i = layer index
    uint j = 0; //current layer node index
    uint k = 0; //previous layer node index

    //find correct pred for softmax
    float * temp = new float[m_pLayer->get_num_nodes()];

    // m_pLossFunc->get_loss_func_derv(pfExpOut, temp);
    lossFunc->get_loss_func_derv(pfExpOut, temp);
    m_pLayer->get_delta_all_nodes(temp);

    for(j = 0; j < m_pLayer->get_num_nodes(); j++)
    {
        m_pLayer->set_node_delta(temp[j], j);
    }

    delete [] temp;
}

void OpenclAccelerator::do_backwardpass_dense_layer()
{
    //i = layer index
    uint j = 0; //current layer node index
    uint k = 0; //previous layer node index

    //find correct pred for softmax
    float * temp = new float[m_pLayer->get_num_nodes()];

    //********************
    // for(j = 0; j < m_pNN->GetLayer(i)->get_num_nodes(); j++)
    // {
    //     temp[j] = 0.0f;

    //     for(k = 0; k < m_pNN->GetLayer(i+1)->get_num_nodes(); k++)
    //     {
    //         temp[j] += m_pNN->GetLayer(i+1)->get_node_delta_idx(k) * m_pNN->GetMatrix(i)->get_weight(j, k);
    //     }
    // }

    //[(unInIdx * m_pOutputLayer->get_num_nodes()) + unOutIdx]

    const float * wtMtx = m_pLayer->get_transform_matrix();
    uint out_lyr_sz = m_pLayer->GetNextLayer()->get_num_nodes();
    uint in_lyr_sz = m_pLayer->get_num_nodes();
    float* outLyrNodeDelta = new float[out_lyr_sz];
    // float * wtNodeProd = new float [in_lyr_sz * out_lyr_sz];
    for(k = 0; k < out_lyr_sz; k++)
    {
        outLyrNodeDelta[k] = m_pLayer->GetNextLayer()->get_node_delta_idx(k);
    }

    // Create buffers
    cl::Buffer wtMtxBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * in_lyr_sz * out_lyr_sz, (void *)wtMtx);
    cl::Buffer outLyrNodeDeltaBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * out_lyr_sz, outLyrNodeDelta);
    cl::Buffer wtNodeProdBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * in_lyr_sz * out_lyr_sz);
    cl::Buffer tempBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * in_lyr_sz, temp);

    // Set kernel arguments
    computeBwWtNodeProdKernel.setArg(0, wtMtxBuffer);
    computeBwWtNodeProdKernel.setArg(1, outLyrNodeDeltaBuffer);
    computeBwWtNodeProdKernel.setArg(2, wtNodeProdBuffer);
    computeBwWtNodeProdKernel.setArg(3, in_lyr_sz);
    computeBwWtNodeProdKernel.setArg(4, out_lyr_sz);

    computeTempDeltaKernel.setArg(0, wtNodeProdBuffer);
    computeTempDeltaKernel.setArg(1, tempBuffer);
    computeTempDeltaKernel.setArg(2, in_lyr_sz);
    computeTempDeltaKernel.setArg(3, out_lyr_sz);

    // Execute kernels
    cl::NDRange global(in_lyr_sz, out_lyr_sz);
    queue.enqueueNDRangeKernel(computeBwWtNodeProdKernel, cl::NullRange, global);
    queue.finish(); // Ensure the first kernel completes before the second kernel starts

    cl::NDRange globalSigma(in_lyr_sz);
    queue.enqueueNDRangeKernel(computeTempDeltaKernel, cl::NullRange, globalSigma);
    queue.finish(); // Ensure the second kernel completes before reading the buffer

    // Read back results
    queue.enqueueReadBuffer(tempBuffer, CL_TRUE, 0, sizeof(float) * in_lyr_sz, temp);

    m_pLayer->get_delta_all_nodes(temp);
    for(j = 0; j < m_pLayer->get_num_nodes(); j++)
    {
        m_pLayer->set_node_delta(temp[j], j);
    }

    delete [] outLyrNodeDelta;
    wtMtxBuffer = nullptr;
    outLyrNodeDeltaBuffer = nullptr;
    wtNodeProdBuffer = nullptr;
    tempBuffer = nullptr;

    delete [] temp;
}
