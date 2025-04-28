#ifndef BINARY_CROSS_ENTROPY_LOSS_H
#define BINARY_CROSS_ENTROPY_LOSS_H

#include "baseLossFunction.h"

class BinaryCrossEntropyLoss : public BaseLossFunction
{
    public:
    BinaryCrossEntropyLoss(nn_layer* nn_lyr);

    float apply_loss_func(float* fExpOut);

    void get_loss_func_derv(float* fExpOut, float* fVal);

    ~BinaryCrossEntropyLoss();

};

#endif