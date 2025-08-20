#include "baseOptimizer.h"
#include "nn_core.h"




void BaseOptimizer::correct_transform_parameters()
{
    uint i; //in layer index, out layer index is always in layer index + 1
    
    

    for(i = 1; i < m_pNN->GetNumLys(); i++)
    {
        switch(m_pNN->get_layer_type(i))
        {
            case eLayer_type::CONV:
                correct_transform_parameters_conv(i);
                break;
            case eLayer_type::POOLING:
                //not required for pooling layer
                break;
            case eLayer_type::DENSE:
                correct_transform_parameters_dense(i);
                break;
            default:
                assert(0); //unknown layer type
                break;
            
        }
    }
}

void BaseOptimizer::correct_transform_parameters_and_biases()
{
    correct_transform_parameters();
    correct_biases();
}