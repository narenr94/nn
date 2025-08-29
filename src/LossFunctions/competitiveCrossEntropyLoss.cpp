#include "competitiveCrossEntropyLoss.h"
#include "baseLayer.h"
#include <cmath>

#define EPSILON 0.000001f

CompetitiveCrossEntropyLoss::CompetitiveCrossEntropyLoss(BaseLayer* nn_lyr)
{
    m_pLayer = nn_lyr;
    m_unOutputLyrSz = m_pLayer->get_num_nodes();
}

float CompetitiveCrossEntropyLoss::apply_loss_func(std::vector<float>& fExpOut)
{
    float fRet = 0.0f;
    float temp = 0.0f;

    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        //loss += actual[i] * std::log(predicted[i]);
        temp = fExpOut[i] * std::log(m_pLayer->get_node_value_idx(i));
        fRet += temp;
    }

    return (-1.0f * (fRet/m_unOutputLyrSz));

    
}

void CompetitiveCrossEntropyLoss::get_loss_func_derv(std::vector<float>& fExpOut, std::vector<float>& fVal)
{
    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        if(m_pLayer->get_node_value_idx(i) == 0.0f)
        {
            m_pLayer->set_node_value(EPSILON, i);
        }
        fVal[i] = ((-1.0f * fExpOut[i]) / (m_pLayer->get_node_value_idx(i)));
    }

}

CompetitiveCrossEntropyLoss::~CompetitiveCrossEntropyLoss(){}