#ifndef MEAN_SQUARED_ERROR_H
#define MEAN_SQUARED_ERROR_H

#include "lossFunction.h"

class MeanSquaredError : public LossFunction
{
    public:
    MeanSquaredError(NeuralNet* nn);

    float apply_loss_func(float* fExpOut);

    float apply_loss_func_derv(float fExpOut, uint idx);

    ~MeanSquaredError();

};

#endif