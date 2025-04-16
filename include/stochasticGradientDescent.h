#ifndef SGD_H
#define SGD_H

#include "baseOptimizer.h"


class StochasticGradientDescent : public BaseOptimizer
{
    private:
        void correct_biases();

        void correct_weights();

    public:

        StochasticGradientDescent(NeuralNet* nn){ m_pNN = nn;}

        void correct_weights_biases();   

        ~StochasticGradientDescent(){}
};


#endif