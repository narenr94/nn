#include "nn_node.h"

nn_node::nn_node()
{
    m_fValue = 0.0;
    m_fBias = 0.0;

    m_fDelta = 0.0;
}

void nn_node::set_bias(float fVal)
{
    NNLOG_TRACE("entering bias:%f", fVal);
    m_fBias = fVal;
    NNLOG_TRACE("exiting");
}

void nn_node::set_value(float fVal)
{
    NNLOG_TRACE("entering n:%f", fVal);    
    m_fValue = fVal;    
    NNLOG_TRACE("exiting");
}

void nn_node::set_delta(float fVal)
{
    NNLOG_TRACE("entering n:%f", fVal);    
    m_fDelta = fVal;    
    NNLOG_TRACE("exiting");
}

float nn_node::get_delta()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return m_fDelta;
}

float nn_node::get_bias()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return m_fBias;
}

float nn_node::get_value()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return m_fValue;    
}
