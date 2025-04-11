#ifndef ACTFUNC_H
#define ACTFUNC_H

#include "nn_math.h"

class nn_layer;

class ActivationFunction{

protected:
    nn_layer * m_pNN_Layer;

public:

    ActivationFunction(){}

    virtual void apply_act_func() = 0;

    virtual void get_delta(float *fVal) = 0; //fVal is either loss func derv (in case output layer) or sum of product of next layer deltas and respective weights (for hidden layer)

    virtual ~ActivationFunction(){}

};

#endif