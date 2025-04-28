#ifndef MEAN_ABSOLUTE_ERROR_H
#define MEAN_ABSOLUTE_ERROR_H

#include "baseLossFunction.h"

class MeanAbsoluteError : public BaseLossFunction
{
    public:
    MeanAbsoluteError(nn_layer* nn_lyr);

    float apply_loss_func(float* fExpOut);

    void get_loss_func_derv(float* fExpOut, float* fVal);

    ~MeanAbsoluteError();

};

#endif