#include "nn_layer.h"


uint nn_layer::get_num_nodes()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return m_unNumNodes;
    
}


nn_layer::~nn_layer()
{
    NNLOG_TRACE("entering");
    if(m_ppNodes)
    {
        free(m_ppNodes);
    }
    NNLOG_TRACE("exiting");
}

nn_layer::nn_layer(uint unNumNodesnodes)
{
    NNLOG_TRACE("entering n_nodes:%d", unNumNodesnodes);
    m_unNumNodes = unNumNodesnodes;

    m_ppNodes = (nn_node**)malloc(m_unNumNodes*sizeof(nn_node));

    uint i = 0;

    for(i = 0; i < m_unNumNodes; i++)
    {
        m_ppNodes[i] = new nn_node();
    }

    m_bInitialized = true;
    NNLOG_TRACE("exiting");
}

bool nn_layer::set_node_value(float fVal, uint unIdx)
{
    NNLOG_TRACE("entering val:%f idx:%d", fVal, unIdx);
    bool bRet = false;

    if(unIdx < m_unNumNodes && m_bInitialized)
    {
        m_ppNodes[unIdx]->set_value(fVal);

        bRet = true;
    }

    NNLOG_TRACE("exiting");

    return bRet;
}

bool nn_layer::set_all_node_values(float* pfValue)
{
    NNLOG_TRACE("entering");
    bool bRet = false;

    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting not initialized!!");
        return bRet;
    }

    bRet = true;

    uint i = 0;

    for(i = 0; i < m_unNumNodes; i++)
    {
        m_ppNodes[i]->set_value(pfValue[i]);
        //NNLOG_MIL("i:%d node[i]->get_value():%f value[i]:%f", i, nodes[i]->get_value(), value[i]);
    }

    NNLOG_TRACE("exiting");

    return bRet;
}

bool nn_layer::set_all_node_biases(float* pfBias)
{
    NNLOG_TRACE("entering");
    bool bRet = false;

    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting not initialized!!!");
        return bRet;
    }

    bRet = true;

    uint i = 0;

    for(i = 0; i < m_unNumNodes; i++)
    {
        m_ppNodes[i]->set_bias(pfBias[i]);
    }

    NNLOG_TRACE("exiting");

    return bRet;

}

bool nn_layer::set_node_bias(float fBias, uint unIdx)
{
    NNLOG_TRACE("entering bias:%f idx:%d", fBias, unIdx);
    bool bRet = false;

    if((unIdx < m_unNumNodes) && m_bInitialized)
    {
        m_ppNodes[unIdx]->set_bias(fBias);

        bRet = true;
    }

    NNLOG_TRACE("exiting");

    return bRet;
}

bool nn_layer::set_node_delta(float fDelta, uint unIdx)
{
    NNLOG_TRACE("entering delta:%f idx:%d", fDelta, unIdx);
    bool bRet = false;

    if((unIdx < m_unNumNodes) && m_bInitialized)
    {
        m_ppNodes[unIdx]->set_delta(fDelta);

        bRet = true;
    }

    NNLOG_TRACE("exiting");

    return bRet;
}

bool nn_layer::is_initialized()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return m_bInitialized;
}

float nn_layer::get_node_value_idx(uint unIdx)
{
    NNLOG_TRACE("entering idx:%d", unIdx);
    NNLOG_TRACE("exiting");
    return m_ppNodes[unIdx]->get_value();
}

float nn_layer::get_node_bias_idx(uint unIdx)
{
    NNLOG_TRACE("entering idx:%d", unIdx);
    NNLOG_TRACE("exiting");
    return m_ppNodes[unIdx]->get_bias();
}

float nn_layer::get_node_delta_idx(uint unIdx)
{
    NNLOG_TRACE("entering idx:%d", unIdx);
    NNLOG_TRACE("exiting");
    return m_ppNodes[unIdx]->get_delta();
}

void nn_layer::set_layer_type(eLyr_type eLType)
{
    NNLOG_TRACE("entering lt:%d", eLType);
    eLyrType = eLType;
    NNLOG_TRACE("exiting");
}

eLyr_type nn_layer::get_layer_type()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return eLyrType;

}

void nn_layer::populateBiasesWithRandomNumbers()
{
    NNLOG_TRACE("entering");

    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting not initialized!!!");
        return;
    }

    uint i = 0;
    float tmp;

    for(i = 0; i < m_unNumNodes; i++)
    {
        tmp = (float)getRandomNumber(RAND_MIN_WEIGHT_BIAS, RAND_MAX_WEIGHT_BIAS);
        tmp /= 10.0;
        m_ppNodes[i]->set_bias(tmp);
    }

    NNLOG_TRACE("exiting");

}
