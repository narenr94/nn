#ifndef NN_OPENCL_ACC_H
#define NN_OPENCL_ACC_H
#include "baseAccelerator.h"
#define CL_HPP_TARGET_OPENCL_VERSION 300
//#define CL_HPP_ENABLE_EXCEPTIONS
#include "CL/opencl.hpp"
#include <iostream>
#include <vector>

class OpenclAccelerator : public BaseAccelerator{

    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel computeWtNodeProdKernel, computeSigmaKernel, computeBwWtNodeProdKernel, computeTempDeltaKernel;

public:

    OpenclAccelerator(BaseLayer* pLayer);

    void do_forwardpass_dense_layer() override;
    void do_backwardpass_dense_layer_output_layer(float* pfExpOut, BaseLossFunction* lossFunc) override;
    void do_backwardpass_dense_layer() override;


   ~OpenclAccelerator();

};
#endif