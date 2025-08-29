#ifndef MEAN_ABSOLUTE_ERROR_H
#define MEAN_ABSOLUTE_ERROR_H

#include "baseLossFunction.h"

class MeanAbsoluteError : public BaseLossFunction
{
    public:
    MeanAbsoluteError(BaseLayer* nn_lyr);

    float apply_loss_func(std::vector<float>& fExpOut);

    void get_loss_func_derv(std::vector<float>& fExpOut, std::vector<float>&fVal);

    ~MeanAbsoluteError();

};

#endif