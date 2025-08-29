#ifndef RELUACTFUNC_H
#define RELUACTFUNC_H

#include "baseActivationFunction.h"

class ReluActFunc : public BaseActivationFunction
{
    public:

        ReluActFunc(BaseLayer* pLayer){m_pLayer = pLayer;}

        void apply_act_func();

        void get_delta(std::vector<float>& fVal);

        ~ReluActFunc(){}

};

#endif