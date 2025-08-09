#ifndef NN_BASE_ACC_H
#define NN_BASE_ACC_H

#include "baseLossFunction.h"

#include "nn_defines.h"

class BaseLayer;

class BaseAccelerator{

protected:

    BaseLayer* m_pLayer;

public:

    BaseAccelerator(BaseLayer* pLayer){}

    //dense layer
    virtual void do_forwardpass_dense_layer() = 0;
    virtual void do_backwardpass_from_output_layer(float* pfExpOut, BaseLossFunction* lossFunc) = 0;
    virtual void do_backwardpass_dense_layer() = 0;

    //convolution layer
    virtual void do_forwardpass_conv_layer(sLayer_Dimensions t_tims) = 0;
    virtual void do_backwardpass_conv_layer(sLayer_Dimensions t_tims) = 0;

    //pooling layer
    virtual void do_forwardpass_pooling_layer(ePooling_type t_pooling_type, ePoolingKernelSize t_pooling_kernel_sz, uint t_stride) = 0;
    // virtual void do_backwardpass_pooling_layer(sLayer_Dimensions t_tims) = 0;

    virtual ~BaseAccelerator(){}

};
#endif