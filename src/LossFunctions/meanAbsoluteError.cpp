#include "meanAbsoluteError.h"
#include "baseLayer.h"
#include <cmath>


MeanAbsoluteError::MeanAbsoluteError(BaseLayer* nn_lyr)
{
    m_pLayer = nn_lyr;
    m_unOutputLyrSz = m_pLayer->get_num_nodes();
}

float MeanAbsoluteError::apply_loss_func(float* fExpOut)
{
    float fRet = 0.0f;
    float temp = 0.0f;

    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        temp = std::fabs(fExpOut[i] - m_pLayer->get_node_value_idx(i));
        fRet += temp;
    }

    return (fRet/m_unOutputLyrSz);
}

void MeanAbsoluteError::get_loss_func_derv(float* fExpOut, float* fVal)
{
    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        if (m_pLayer->get_node_value_idx(i) > fExpOut[i]) 
        { 
            fVal[i] = 1.0f / m_unOutputLyrSz; 
            // fRet = 1.0f;
        } else if (m_pLayer->get_node_value_idx(i) < fExpOut[i]) 
        { 
            fVal[i] = -1.0f / m_unOutputLyrSz; 
            // fRet = -1.0f;
        } else 
        { 
            fVal[i] = 0.0f; 
        }
    }

}

MeanAbsoluteError::~MeanAbsoluteError(){}