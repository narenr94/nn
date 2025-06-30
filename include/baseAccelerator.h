#ifndef NN_BASE_ACC_H
#define NN_BASE_ACC_H
#include "nn_math.h"
#include "baseLossFunction.h"

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
    virtual void do_forwardpass_conv_layer(uint input_rows, uint input_columns, uint filter_rows, uint filter_columns) = 0;
    virtual void do_backwardpass_conv_layer(uint input_rows, uint input_columns, uint filter_rows, uint filter_columns) = 0;

    virtual ~BaseAccelerator(){}

};
#endif