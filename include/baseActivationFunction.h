#ifndef BASE_ACTFUNC_H
#define BASE_ACTFUNC_H

#include "nn_math.h"

class BaseLayer;

class BaseActivationFunction{

protected:
    BaseLayer * m_pLayer;

public:

    BaseActivationFunction(){}

    virtual void apply_act_func() = 0;

    virtual void get_delta(float *fVal) = 0; //fVal is either loss func derv (in case output layer) or sum of product of next layer deltas and respective weights (for hidden layer)

    virtual ~BaseActivationFunction(){}

};

#endif