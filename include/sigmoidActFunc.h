#ifndef SIGMOIDACTFUNC_H
#define SIGMOIDACTFUNC_H

#include "baseActivationFunction.h"

class SigmoidActFunc : public BaseActivationFunction
{

    public:

        SigmoidActFunc(BaseLayer* pLayer){m_pLayer = pLayer;}

        void apply_act_func();

        void get_delta(std::vector<float>& fVal);

        ~SigmoidActFunc(){}

};

#endif