#ifndef HUBER_LOSS_H
#define HUBER_LOSS_H

#include "lossFunction.h"

class HuberLoss : public LossFunction
{
    private:
        float m_fDelta;

    public:
    HuberLoss(NeuralNet* nn, float delta = 1.0f);

    float apply_loss_func(float* fExpOut);

    float apply_loss_func_derv(float fExpOut, uint idx);

    ~HuberLoss();

};

#endif