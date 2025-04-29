#include "sigmoidActFunc.h"
#include "baseLayer.h"

void SigmoidActFunc::apply_act_func()
{
    for(uint i = 0; i < m_pLayer->get_num_nodes(); i++)
    {
        m_pLayer->set_node_value(get_sigmoidf(m_pLayer->get_node_value_idx(i)), i);
    }
}

void SigmoidActFunc::get_delta(float* fVal)
{
    for(uint i = 0; i < m_pLayer->get_num_nodes(); i++)
    {
        fVal[i] *= find_derivative_sigmoidf(m_pLayer->get_node_value_idx(i));
    }
}