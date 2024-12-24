#ifndef RELUACTFUNC_H
#define RELUACTFUNC_H

#include "activationFunction.h"

class ReluActFunc : public ActivationFunction
{
    public:

        ReluActFunc(){}

        float apply_act_func(float n);

        float apply_act_func_derv(float fVal);

        ~ReluActFunc(){}

};

#endif