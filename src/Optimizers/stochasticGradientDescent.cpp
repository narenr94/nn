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
            if(m_pNN->GetLayer(i)->get_layer_type() == eLayer_type::POOLING)
            {
                continue;
            }
            m_pNN->SetBias(i, j, m_pNN->GetBias(i, j) - (m_pNN->GetLearningRate() * m_pNN->GetDelta(i, j)));
        }        
    }
    
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

    uint kernelSz = k_rows * k_cols;

    // uint out_rows = layer_dim.unOutputRows;
    // uint out_cols = layer_dim.unOutputColumns;

    uint out_rows = layer_dim.unInputRows - layer_dim.unTransformParametersRows + 1;
    uint out_cols = layer_dim.unInputColumns - layer_dim.unTransformParametersColumns + 1;

    uint outSz = out_rows * out_cols;

    uint curr_idx = 0;
    uint prev_idx = 0;

    for(uint f = 0; f < layer_dim.unNoTransformParameterMtx; f++)
    {

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
                        curr_idx += (f * outSz);
                        prev_idx = ((p + l) * (layer_dim.unInputColumns)) + (q + m);
                        delta_wt += m_pNN->GetDelta(i, curr_idx) * m_pNN->GetNodeVal(i - 1, prev_idx);
                    }
                }
                delta_wt *= m_pNN->GetLearningRate();
                uint mtx_idx = (l * k_cols) + m;
                mtx_idx += (f * kernelSz);
                // dW[i][j] = delta_wt;
                m_pNN->SetWeight(i, mtx_idx, (m_pNN->GetWeight(i, mtx_idx) - delta_wt));
            }
        }
    }

}
