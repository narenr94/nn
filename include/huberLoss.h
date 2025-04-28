#ifndef HUBER_LOSS_H
#define HUBER_LOSS_H

#include "baseLossFunction.h"

#define HUBER_DEFAULT_DELTA 1.0f

class HuberLoss : public BaseLossFunction
{
    private:
        float m_fDelta;

    public:
    HuberLoss(nn_layer* nn_lyr, float delta = HUBER_DEFAULT_DELTA);

    float apply_loss_func(float* fExpOut);

    void get_loss_func_derv(float* fExpOut, float* fVal);

    ~HuberLoss();

};

#endif