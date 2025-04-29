#ifndef TANHACTFUNC_H
#define TANHACTFUNC_H

#include "baseActivationFunction.h"

class TanhActFunc : public BaseActivationFunction
{

    public:

        TanhActFunc(BaseLayer* pLayer){m_pLayer = pLayer;}

        void apply_act_func();

        void get_delta(float* fVal);

        ~TanhActFunc(){}

};

#endif