#ifndef COMPETITIVE_CROSS_ENTROPY_LOSS_H
#define COMPETITIVE_CROSS_ENTROPY_LOSS_H

#include "baseLossFunction.h"

class CompetitiveCrossEntropyLoss : public BaseLossFunction
{
    public:
    CompetitiveCrossEntropyLoss(nn_layer* nn_lyr);

    float apply_loss_func(float* fExpOut);

    void get_loss_func_derv(float* fExpOut, float* fVal);

    ~CompetitiveCrossEntropyLoss();

};

#endif