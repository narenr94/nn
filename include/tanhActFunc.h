#ifndef TANHACTFUNC_H
#define TANHACTFUNC_H

#include "activationFunction.h"

class TanhActFunc : public ActivationFunction
{

    public:

        TanhActFunc(){}

        float apply_act_func(float n);

        float apply_act_func_derv(float fVal);

        ~TanhActFunc(){}

};

#endif