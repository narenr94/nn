#include "openclAccelerator.h"
#include "nn_core.h"

OpenclAccelerator::OpenclAccelerator(NeuralNet* pNN) : BaseAccelerator(pNN)
{
    cl_int err;
    err = clGetPlatformIDs(1, &platformId, NULL);
    assert(err == CL_SUCCESS && "Error getting platform ID");
    err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, NULL);
    assert(err == CL_SUCCESS && "Error getting device ID");
    context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &err);
    assert(err == CL_SUCCESS && "Error creating context");
    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, 0, 0 };
    queue = clCreateCommandQueueWithProperties(context, deviceId, properties, &err);
    assert(err == CL_SUCCESS && "Error creating command queue");

    /*
    // Print platform and device details
    char platformName[128];
    char deviceName[128];
    err = clGetPlatformInfo(platformId, CL_PLATFORM_NAME, 128, platformName, NULL);
    assert(err == CL_SUCCESS && "Error getting platform name");
    printf("Platform: %s\n", platformName);

    err = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 128, deviceName, NULL);
    assert(err == CL_SUCCESS && "Error getting device name");
    printf("Device: %s\n", deviceName);
    */

}

OpenclAccelerator::~OpenclAccelerator()
{
    // Release resources
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

// void OpenclAccelerator::do_forwardpass_to_next_layer(uint unInLayerIdx)
// {
    
// }

// void OpenclAccelerator::find_delta_of_all_nodes(float* pfExpOut)
// {
    

// }