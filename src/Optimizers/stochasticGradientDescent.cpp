#include "stochasticGradientDescent.h"
#include "nn_core.h"


void StochasticGradientDescent::correct_biases()
{
    uint i; //in layer index, out layer index is always in layer index + 1
    uint j; //in layer node index

    for(i = 0; i < (m_pNN->GetNumLys() - 1); i++)
    {
        for(j = 0; j < m_pNN->GetSzLayer(i); j++)
        {
            m_pNN->SetBias(i, j, m_pNN->GetBias(i, j) - (m_pNN->GetLearningRate() * m_pNN->GetDelta(i, j)));
            
        }
    }
}

void StochasticGradientDescent::correct_weights()
{
    uint i; //in layer index, out layer index is always in layer index + 1
    uint j; //in layer node index
    uint k; //out layer node index

    float delta_wt = 0.0;

    for(i = 0; i < m_pNN->GetNumLys() - 1; i++)
    {
        for(j = 0; j < m_pNN->GetSzLayer(i); j++)
        {
            for(k = 0; k < m_pNN->GetSzLayer(i+1); k++)
            {

                delta_wt = m_pNN->GetDelta(i + 1, k) * m_pNN->GetNodeVal(i, j);
                delta_wt *= m_pNN->GetLearningRate();
                m_pNN->SetWeight(i, j, k, (m_pNN->GetWeight(i, j, k) - delta_wt));
            }
            

        }
    }
}

void StochasticGradientDescent::correct_weights_biases()
{
    correct_weights();
    correct_biases();
}

