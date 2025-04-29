#include "meanSquaredError.h"
#include "baseLayer.h"
#include <cmath>
#include <cstdio>

MeanSquaredError::MeanSquaredError(BaseLayer* nn_lyr)
{
    m_pLayer = nn_lyr;
    m_unOutputLyrSz = m_pLayer->get_num_nodes();
}

float MeanSquaredError::apply_loss_func(float* fExpOut)
{
    float fRet = 0.0f;
    float temp = 0.0f;

    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        temp = fExpOut[i] - m_pLayer->get_node_value_idx(i);
        temp *= temp;
        fRet += temp;
    }

    return (fRet/m_unOutputLyrSz);
}

void MeanSquaredError::get_loss_func_derv(float* fExpOut, float* fVal)
{
    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        fVal[i] = (-2.0f * ((fExpOut[i] - m_pLayer->get_node_value_idx(i)) / m_unOutputLyrSz));
    }

}

MeanSquaredError::~MeanSquaredError(){}