#ifndef NN_CPU_ACC_H
#define NN_CPU_ACC_H
#include "baseAccelerator.h"


class CpuAccelerator : public BaseAccelerator{


public:

    CpuAccelerator(BaseLayer* pNLyr);

    void do_forwardpass_dense_layer() override;
    void do_backwardpass_from_output_layer(float* pfExpOut, BaseLossFunction* lossFunc) override;
    void do_backwardpass_dense_layer() override;

    void do_forwardpass_conv_layer(sLayer_Dimensions t_tims) override;
    void do_backwardpass_conv_layer(sLayer_Dimensions t_tims) override;

    virtual ~CpuAccelerator();

};
#endif