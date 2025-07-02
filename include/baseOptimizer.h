#ifndef BASE_OPTIMIZER_H
#define BASE_OPTIMIZER_H

#include "nn_math.h"

class NeuralNet; //forward declaration, class defined in nn_core.h


class BaseOptimizer{

protected:

NeuralNet* m_pNN;

public:

    BaseOptimizer(){}

    virtual void correct_transform_parameters_and_biases() = 0;

    virtual ~BaseOptimizer(){}

private:

    virtual void correct_transform_parameters_dense(uint curr_lyr_idx) = 0;

    virtual void correct_transform_parameters_conv(uint curr_lyr_idx) = 0;

};

#endif