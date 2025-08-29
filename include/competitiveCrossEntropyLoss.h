#ifndef COMPETITIVE_CROSS_ENTROPY_LOSS_H
#define COMPETITIVE_CROSS_ENTROPY_LOSS_H

#include "baseLossFunction.h"

class CompetitiveCrossEntropyLoss : public BaseLossFunction
{
    public:
    CompetitiveCrossEntropyLoss(BaseLayer* nn_lyr);

    float apply_loss_func(std::vector<float>& fExpOut);

    void get_loss_func_derv(std::vector<float>& fExpOut, std::vector<float>& fVal);

    ~CompetitiveCrossEntropyLoss();

};

#endif