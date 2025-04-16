#ifndef RELUACTFUNC_H
#define RELUACTFUNC_H

#include "baseActivationFunction.h"

class ReluActFunc : public BaseActivationFunction
{
    public:

        ReluActFunc(nn_layer* pNN_Layer){m_pNN_Layer = pNN_Layer;}

        void apply_act_func();

        void get_delta(float* fVal);

        ~ReluActFunc(){}

};

#endif