#include "binaryCrossEntropyLoss.h"
#include "nn_layer.h"
#include <cmath>

#define EPSILON 0.000001f

BinaryCrossEntropyLoss::BinaryCrossEntropyLoss(nn_layer* nn_lyr)
{
    m_pLayer = nn_lyr;
    m_unOutputLyrSz = m_pLayer->get_num_nodes();
}

float BinaryCrossEntropyLoss::apply_loss_func(float* fExpOut)
{
    float fRet = 0.0f;
    float temp = 0.0f;

    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        //ce += y_true[i] * std::log(y_pred[i]) + (1 - y_true[i]) * std::log(1 - y_pred[i]);
        temp = (fExpOut[i] * log(m_pLayer->get_node_value_idx(i))) + ((1.0f - fExpOut[i]) * std::log(1.0f - m_pLayer->get_node_value_idx(i)));
        fRet += temp;
    }

    return (-1.0f * (fRet/m_unOutputLyrSz));
}

void BinaryCrossEntropyLoss::get_loss_func_derv(float* fExpOut, float* fVal)
{
    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        if(m_pLayer->get_node_value_idx(i) == 0.0f)
        {
            m_pLayer->set_node_value(EPSILON, i);
        }
        fVal[i] = ((m_pLayer->get_node_value_idx(i) - fExpOut[i]) / (m_pLayer->get_node_value_idx(i) * (1.0f - m_pLayer->get_node_value_idx(i))));
        
    }

}

BinaryCrossEntropyLoss::~BinaryCrossEntropyLoss(){}