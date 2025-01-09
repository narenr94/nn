#ifndef LEAKYRELUACTFUNC_H
#define LEAKYRELUACTFUNC_H

#include "activationFunction.h"

#define LEAKY_RELU_DEFAULT_ALPHA 0.01f

class LeakyReluActFunc : public ActivationFunction
{

    float m_alpha;

    public:

        LeakyReluActFunc(nn_layer* pNN_Layer, float alpha = LEAKY_RELU_DEFAULT_ALPHA){m_pNN_Layer = pNN_Layer, m_alpha = alpha;}

        void apply_act_func();

        void get_delta(float* fVal);

        ~LeakyReluActFunc(){}

};

#endif