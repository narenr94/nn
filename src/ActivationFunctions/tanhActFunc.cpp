#include "tanhActFunc.h"
#include "baseLayer.h"

void TanhActFunc::apply_act_func()
{
    for(uint i = 0; i < m_pLayer->get_num_nodes(); i++)
    {
        m_pLayer->set_node_value(get_tanhf(m_pLayer->get_node_value_idx(i)), i);
    }
}

void TanhActFunc::get_delta(std::vector<float>& fVal)
{
    for(uint i = 0; i < m_pLayer->get_num_nodes(); i++)
    {
        fVal[i] *= find_derivative_tanhf(m_pLayer->get_node_value_idx(i));
    }
}