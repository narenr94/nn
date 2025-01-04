#ifndef LEAKYRELUACTFUNC_H
#define LEAKYRELUACTFUNC_H

#include "activationFunction.h"

#define LEAKY_RELU_DEFAULT_ALPHA 0.01f

class LeakyReluActFunc : public ActivationFunction
{

    float m_alpha;

    public:

        LeakyReluActFunc(float alpha = LEAKY_RELU_DEFAULT_ALPHA){m_alpha = alpha;}

        float apply_act_func(float n);

        float apply_act_func_derv(float fVal);

        ~LeakyReluActFunc(){}

};

#endif