#ifndef BASE_LOSSFUNC_H
#define BASE_LOSSFUNC_H

#include "nn_math.h"

class BaseLayer; //forward declaration, class defined in baseLayer.h

class BaseLossFunction{

protected:

    BaseLayer* m_pLayer; //output layer

    uint m_unOutputLyrSz; //output layer size

public:

    BaseLossFunction(){}

    virtual float apply_loss_func(float* fExpOut) = 0;

    virtual void get_loss_func_derv(float* fExpOut, float* fVal) = 0;

    virtual ~BaseLossFunction(){}

};

#endif