#include "leakyReluActFunc.h"
#include "nn_layer.h"

void LeakyReluActFunc::apply_act_func()
{
    for(uint i = 0; i < m_pNN_Layer->get_num_nodes(); i++)
    {
        m_pNN_Layer->set_node_value(get_leakyReluf(m_pNN_Layer->get_node_value_idx(i ), m_alpha), i);
    }
    
}

void LeakyReluActFunc::get_delta(float* fVal)
{
    for(uint i = 0; i < m_pNN_Layer->get_num_nodes(); i++)
    {
        fVal[i] *= find_derivative_leakyReluf(m_pNN_Layer->get_node_value_idx(i), m_alpha);
    }
}