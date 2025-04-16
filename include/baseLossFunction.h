#ifndef BASE_LOSSFUNC_H
#define BASE_LOSSFUNC_H

#include "nn_math.h"

class NeuralNet; //forward declaration, class defined in nn_core.h

class BaseLossFunction{

protected:

    NeuralNet* m_pNN;

    uint m_unOutputLyrSz;

    uint m_unOutputLyrID;

public:

    BaseLossFunction(){}

    virtual float apply_loss_func(float* fExpOut) = 0;

    virtual void get_loss_func_derv(float* fExpOut, float* fVal) = 0;

    virtual ~BaseLossFunction(){}

};

#endif