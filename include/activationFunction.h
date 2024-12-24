#ifndef ACTFUNC_H
#define ACTFUNC_H

#include "nn_math.h"

class ActivationFunction{

public:

    ActivationFunction(){}

    virtual float apply_act_func(float n);

    virtual float apply_act_func_derv(float fVal);

    virtual ~ActivationFunction(){}

};

#endif