#ifndef NN_OPENCL_ACC_H
#define NN_OPENCL_ACC_H
#include "baseAccelerator.h"
#define CL_HPP_TARGET_OPENCL_VERSION 300
//#define CL_HPP_ENABLE_EXCEPTIONS
#include "CL/opencl.hpp"
#include <iostream>
#include <vector>

class NeuralNet;

class OpenclAccelerator : public BaseAccelerator{

    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel computeWtNodeProdKernel, computeSigmaKernel, computeBwWtNodeProdKernel, computeTempDeltaKernel;

public:

    OpenclAccelerator(NeuralNet* pNN);

    void do_forwardpass_to_next_layer(uint unInLayerIdx) override;
    void find_delta_of_all_nodes(float* pfExpOut) override;


   ~OpenclAccelerator();

};
#endif