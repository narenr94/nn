#ifndef LEAKYRELUACTFUNC_H
#define LEAKYRELUACTFUNC_H

#include "baseActivationFunction.h"

#define LEAKY_RELU_DEFAULT_ALPHA 0.01f

class LeakyReluActFunc : public BaseActivationFunction
{

    float m_alpha;

    public:

        LeakyReluActFunc(BaseLayer* pLayer, float alpha = LEAKY_RELU_DEFAULT_ALPHA){m_pLayer = pLayer, m_alpha = alpha;}

        void apply_act_func();

        void get_delta(std::vector<float>& fVal);

        ~LeakyReluActFunc(){}

};

#endif