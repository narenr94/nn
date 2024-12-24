#ifndef SIGMOIDACTFUNC_H
#define SIGMOIDACTFUNC_H

#include "activationFunction.h"

class SigmoidActFunc : public ActivationFunction
{

    public:

        SigmoidActFunc(){}

        float apply_act_func(float n);

        float apply_act_func_derv(float fVal);

        ~SigmoidActFunc(){}

};

#endif