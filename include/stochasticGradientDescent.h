#ifndef SGD_H
#define SGD_H

#include "baseOptimizer.h"


class StochasticGradientDescent : public BaseOptimizer
{
    private:
        void correct_biases() override;

    public:

        StochasticGradientDescent(NeuralNet* nn){ m_pNN = nn;} 

        ~StochasticGradientDescent(){}

    private:
    
        void correct_transform_parameters_dense(uint curr_lyr_idx) override;

        void correct_transform_parameters_conv(uint curr_lyr_idx) override;
};


#endif