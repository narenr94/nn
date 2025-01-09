#include "tanhActFunc.h"
#include "nn_layer.h"

void TanhActFunc::apply_act_func()
{
    for(uint i = 0; i < m_pNN_Layer->get_num_nodes(); i++)
    {
        m_pNN_Layer->set_node_value(get_tanhf(m_pNN_Layer->get_node_value_idx(i)), i);
    }
}

void TanhActFunc::get_delta(float* fVal)
{
    for(uint i = 0; i < m_pNN_Layer->get_num_nodes(); i++)
    {
        fVal[i] *= find_derivative_tanhf(m_pNN_Layer->get_node_value_idx(i));
    }
}