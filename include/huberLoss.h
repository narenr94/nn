#ifndef HUBER_LOSS_H
#define HUBER_LOSS_H

#include "lossFunction.h"

#define HUBER_DEFAULT_DELTA 1.0f

class HuberLoss : public LossFunction
{
    private:
        float m_fDelta;

    public:
    HuberLoss(NeuralNet* nn, float delta = HUBER_DEFAULT_DELTA);

    float apply_loss_func(float* fExpOut);

    float apply_loss_func_derv(float fExpOut, uint idx);

    ~HuberLoss();

};

#endif