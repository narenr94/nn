#include "reluActFunc.h"
#include "nn_layer.h"

void ReluActFunc::apply_act_func()
{
    for(uint i = 0; i < m_pNN_Layer->get_num_nodes(); i++)
    {
        m_pNN_Layer->set_node_value(get_reluf(m_pNN_Layer->get_node_value_idx(i)), i);
    }
}

void ReluActFunc::get_delta(float* fVal)
{
    for(uint i = 0; i < m_pNN_Layer->get_num_nodes(); i++)
    {
        fVal[i] *= find_derivative_reluf(m_pNN_Layer->get_node_value_idx(i));
    }
}