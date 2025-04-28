#ifndef BASE_LOSSFUNC_H
#define BASE_LOSSFUNC_H

#include "nn_math.h"

/*
    list of activation functions
    Note : keep CCE in bottom to keep tests intact
*/
enum eLossFuncs{
    MSE, //mean squared error
    MAE, //mean absolute error
    HUBER, //huber loss
    BCE, //binary cross entropy loss
    CCE //competitive cross entropy loss
};

class nn_layer; //forward declaration, class defined in nn_layer.h

class BaseLossFunction{

protected:

    nn_layer* m_pLayer; //output layer

    uint m_unOutputLyrSz; //output layer size

public:

    BaseLossFunction(){}

    virtual float apply_loss_func(float* fExpOut) = 0;

    virtual void get_loss_func_derv(float* fExpOut, float* fVal) = 0;

    virtual ~BaseLossFunction(){}

};

#endif