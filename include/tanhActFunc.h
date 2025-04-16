#ifndef TANHACTFUNC_H
#define TANHACTFUNC_H

#include "baseActivationFunction.h"

class TanhActFunc : public BaseActivationFunction
{

    public:

        TanhActFunc(nn_layer* pNN_Layer){m_pNN_Layer = pNN_Layer;}

        void apply_act_func();

        void get_delta(float* fVal);

        ~TanhActFunc(){}

};

#endif