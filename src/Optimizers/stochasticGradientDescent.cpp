#include "stochasticGradientDescent.h"
#include "nn_core.h"
#include <cmath>

#include <stdio.h>


void StochasticGradientDescent::correct_biases()
{
    uint i; //in layer index, out layer index is always in layer index + 1
    uint j; //in layer node index

    for(i = 1; i < m_pNN->GetNumLys(); i++)
    {
        for(j = 0; j < m_pNN->GetSzLayer(i); j++)
        {
            m_pNN->SetBias(i, j, m_pNN->GetBias(i, j) - (m_pNN->GetLearningRate() * m_pNN->GetDelta(i, j)));
        }        
    }
    
}

void StochasticGradientDescent::correct_transform_parameters()
{
    uint i; //in layer index, out layer index is always in layer index + 1
    
    

    for(i = 1; i < m_pNN->GetNumLys(); i++)
    {
        switch(m_pNN->get_layer_type(i))
        {
            case eLayer_type::CONV:
                correct_transform_parameters_conv(i);
                break;
            case eLayer_type::DENSE:
            default:
                correct_transform_parameters_dense(i);
                break;
            
        }
    }
}

void StochasticGradientDescent::correct_transform_parameters_and_biases()
{
    correct_transform_parameters();
    correct_biases();
}


void StochasticGradientDescent::correct_transform_parameters_dense(uint curr_lyr_idx)
{
    uint j; //in layer node index
    uint k; //out layer node index

    float delta_wt = 0.0;

    uint i = curr_lyr_idx;

    for(j = 0; j < m_pNN->GetSzLayer(i - 1); j++)
    {
        for(k = 0; k < m_pNN->GetSzLayer(i); k++)
        {
            delta_wt = m_pNN->GetDelta(i, k) * m_pNN->GetNodeVal(i - 1, j);
            delta_wt *= m_pNN->GetLearningRate();
            uint mtx_idx = (j * m_pNN->GetSzLayer(i)) + k;
            m_pNN->SetWeight(i, j, k, (m_pNN->GetWeight(i, mtx_idx) - delta_wt));
        }            

    }
}

void StochasticGradientDescent::correct_transform_parameters_conv(uint curr_lyr_idx)
{

    uint j; //in layer node index
    uint k; //out layer node index

    float delta_wt = 0.0f;

    uint i = curr_lyr_idx;

    sLayer_Dimensions layer_dim = m_pNN->GetLayer(i)->get_layer_dimensions();

    uint k_rows = layer_dim.unTransformParametersRows;
    uint k_cols = layer_dim.unTransformParametersColumns;

    uint out_rows = layer_dim.unOutputRows;
    uint out_cols = layer_dim.unOutputColumns;

    uint curr_idx = 0;
    uint prev_idx = 0;

    for (int l = 0; l < k_rows; ++l) 
    {
        for (int m = 0; m < k_cols; ++m) 
        {
            delta_wt = 0.0f;

            for (int p = 0; p < out_rows; ++p) 
            {
                for (int q = 0; q < out_cols; ++q) 
                {
                    // delta_wt += dOut[p][q] * input[p + l][q + m];
                    curr_idx = (p * out_cols) + q;
                    prev_idx = ((p + l) * (layer_dim.unInputColumns)) + (q + m);
                    delta_wt += m_pNN->GetDelta(i, curr_idx) * m_pNN->GetNodeVal(i - 1, prev_idx);
                }
            }
            delta_wt *= m_pNN->GetLearningRate();
            uint mtx_idx = (l * m_pNN->GetSzLayer(i)) + m;
            // dW[i][j] = delta_wt;
            m_pNN->SetWeight(i, l, m, (m_pNN->GetWeight(i, mtx_idx) - delta_wt));
        }
    }

}
