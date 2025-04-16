#ifndef SIGMOIDACTFUNC_H
#define SIGMOIDACTFUNC_H

#include "baseActivationFunction.h"

class SigmoidActFunc : public BaseActivationFunction
{

    public:

        SigmoidActFunc(nn_layer* pNN_Layer){m_pNN_Layer = pNN_Layer;}

        void apply_act_func();

        void get_delta(float* fVal);

        ~SigmoidActFunc(){}

};

#endif