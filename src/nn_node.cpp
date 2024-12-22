#include "nn_node.h"

nn_node::nn_node()
{
    m_fValue = 0.0;
    m_fBias = 0.0;

    m_fDelta = 0.0;
}

void nn_node::set_bias(float fVal)
{
    m_fBias = fVal;
}

void nn_node::set_value(float fVal)
{
    m_fValue = fVal;
}

void nn_node::set_delta(float fVal)
{
    m_fDelta = fVal;
}

float nn_node::get_delta()
{
    return m_fDelta;
}

float nn_node::get_bias()
{
    return m_fBias;
}

float nn_node::get_value()
{
    return m_fValue;    
}
