#ifndef SGD_H
#define SGD_H

#include "baseOptimizer.h"


class StochasticGradientDescent : public BaseOptimizer
{
    private:
        void correct_biases();

        void correct_transform_parameters();

    public:

        StochasticGradientDescent(NeuralNet* nn){ m_pNN = nn;}

        void correct_transform_parameters_and_biases();   

        ~StochasticGradientDescent(){}
};


#endif