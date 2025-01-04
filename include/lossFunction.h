#ifndef LOSSFUNC_H
#define LOSSFUNC_H

#include "nn_math.h"

class NeuralNet; //forward declaration, class defined in nn_core.h

class LossFunction{

protected:

    NeuralNet* m_pNN;

    uint m_unOutputLyrSz;

    uint m_unOutputLyrID;

public:

    LossFunction(){}

    virtual float apply_loss_func(float* fExpOut);

    virtual float apply_loss_func_derv(float fExpOut, uint idx);

    virtual ~LossFunction(){}

};

#endif