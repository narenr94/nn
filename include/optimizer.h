#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "nn_math.h"

class NeuralNet; //forward declaration, class defined in nn_core.h


class Optimizer{

protected:

NeuralNet* m_pNN;

public:

    Optimizer(){}

    virtual void correct_weights_biases() = 0;

    virtual ~Optimizer(){}

};

#endif