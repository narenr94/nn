#ifndef SGD_H
#define SGD_H

#include "optimizer.h"


class StochasticGradientDescent : public Optimizer
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