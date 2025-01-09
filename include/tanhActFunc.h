#ifndef TANHACTFUNC_H
#define TANHACTFUNC_H

#include "activationFunction.h"

class TanhActFunc : public ActivationFunction
{

    public:

        TanhActFunc(nn_layer* pNN_Layer){m_pNN_Layer = pNN_Layer;}

        void apply_act_func();

        void get_delta(float* fVal);

        ~TanhActFunc(){}

};

#endif