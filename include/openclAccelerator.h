#ifndef NN_OPENCL_ACC_H
#define NN_OPENCL_ACC_H
#include "baseAccelerator.h"
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

class NeuralNet;

class OpenclAccelerator : public BaseAccelerator{

    cl_platform_id platformId;
    cl_device_id deviceId;
    cl_context context;
    cl_command_queue queue;

public:

    OpenclAccelerator(NeuralNet* pNN);

    // void do_forwardpass_to_next_layer(uint unInLayerIdx) override;
    // void find_delta_of_all_nodes(float* pfExpOut) override;


   ~OpenclAccelerator();

};
#endif