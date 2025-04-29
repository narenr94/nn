#include "reluActFunc.h"
#include "baseLayer.h"

void ReluActFunc::apply_act_func()
{
    for(uint i = 0; i < m_pLayer->get_num_nodes(); i++)
    {
        m_pLayer->set_node_value(get_reluf(m_pLayer->get_node_value_idx(i)), i);
    }
}

void ReluActFunc::get_delta(float* fVal)
{
    for(uint i = 0; i < m_pLayer->get_num_nodes(); i++)
    {
        fVal[i] *= find_derivative_reluf(m_pLayer->get_node_value_idx(i));
    }
}