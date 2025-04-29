#ifndef SOFTMAXACTFUNC_H
#define SOFTMAXACTFUNC_H

#include "baseActivationFunction.h"

class BaseLayer;

class SoftmaxActFunc : public BaseActivationFunction
{
    private:

        BaseLayer* m_pLayer;
        uint m_unCorrectIdx = 0;
        uint m_unSzLyr = 0;
        float m_fDervCorrPred = 0.0f;

        float **m_ppDervMatrix;

        void populate_derv_matrix();

    public:

        SoftmaxActFunc(BaseLayer* pLayer);

        void apply_act_func();

        void get_delta(float* fVal);

        ~SoftmaxActFunc();

};

#endif