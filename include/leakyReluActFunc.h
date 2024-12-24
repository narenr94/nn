#ifndef LEAKYRELUACTFUNC_H
#define LEAKYRELUACTFUNC_H

#include "activationFunction.h"

class LeakyReluActFunc : public ActivationFunction
{

    float m_alpha;

    public:

        LeakyReluActFunc(float alpha = 0.01f){m_alpha = alpha;}

        float apply_act_func(float n);

        float apply_act_func_derv(float fVal);

        ~LeakyReluActFunc(){}

};

#endif