#ifndef MEAN_SQUARED_ERROR_H
#define MEAN_SQUARED_ERROR_H

#include "baseLossFunction.h"

class MeanSquaredError : public BaseLossFunction
{
    public:
    MeanSquaredError(BaseLayer* nn_lyr);

    float apply_loss_func(std::vector<float>& fExpOut);

    void get_loss_func_derv(std::vector<float>& fExpOut, std::vector<float>& fVal);

    ~MeanSquaredError();

};

#endif