#ifndef SIGMOIDACTFUNC_H
#define SIGMOIDACTFUNC_H

#include "activationFunction.h"

class SigmoidActFunc : public ActivationFunction
{

    public:

        SigmoidActFunc(nn_layer* pNN_Layer){m_pNN_Layer = pNN_Layer;}

        void apply_act_func();

        void get_delta(float* fVal);

        ~SigmoidActFunc(){}

};

#endif