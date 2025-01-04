#ifndef MEAN_ABSOLUTE_ERROR_H
#define MEAN_ABSOLUTE_ERROR_H

#include "lossFunction.h"

class MeanAbsoluteError : public LossFunction
{
    public:
    MeanAbsoluteError(NeuralNet* nn);

    float apply_loss_func(float* fExpOut);

    float apply_loss_func_derv(float fExpOut, uint idx);

    ~MeanAbsoluteError();

};

#endif