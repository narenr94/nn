#ifndef SOFTMAXACTFUNC_H
#define SOFTMAXACTFUNC_H

#include "activationFunction.h"

class nn_layer;

class SoftmaxActFunc : public ActivationFunction
{
    private:

        nn_layer* m_pNN_Layer;
        uint m_unCorrectIdx = 0;
        uint m_unSzLyr = 0;
        float m_fDervCorrPred = 0.0f;

        float **m_ppDervMatrix;

        void populate_derv_matrix();

    public:

        SoftmaxActFunc(nn_layer* pNN_Layer);

        void apply_act_func();

        void get_delta(float* fVal);

        ~SoftmaxActFunc();

};

#endif