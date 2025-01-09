#include "sigmoidActFunc.h"
#include "nn_layer.h"

void SigmoidActFunc::apply_act_func()
{
    for(uint i = 0; i < m_pNN_Layer->get_num_nodes(); i++)
    {
        m_pNN_Layer->set_node_value(get_sigmoidf(m_pNN_Layer->get_node_value_idx(i)), i);
    }
}

void SigmoidActFunc::get_delta(float* fVal)
{
    for(uint i = 0; i < m_pNN_Layer->get_num_nodes(); i++)
    {
        fVal[i] *= find_derivative_sigmoidf(m_pNN_Layer->get_node_value_idx(i));
    }
}