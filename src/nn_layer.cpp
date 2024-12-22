#include "nn_layer.h"


uint nn_layer::get_num_nodes()
{
    return m_unNumNodes;
    
}


nn_layer::~nn_layer()
{
    if(m_ppNodes)
    {
        for(uint i = 0; i < m_unNumNodes; i++)
        {
            delete m_ppNodes[i];
        }
        delete [] m_ppNodes;
    }
}

nn_layer::nn_layer(uint unNumNodesnodes)
{
    m_unNumNodes = unNumNodesnodes;

    m_ppNodes = new nn_node*[m_unNumNodes];

    uint i = 0;

    for(i = 0; i < m_unNumNodes; i++)
    {
        m_ppNodes[i] = new nn_node();
    }

    m_bInitialized = true;
}

bool nn_layer::set_node_value(float fVal, uint unIdx)
{
    bool bRet = false;

    if(unIdx < m_unNumNodes && m_bInitialized)
    {
        m_ppNodes[unIdx]->set_value(fVal);

        bRet = true;
    }

    
    return bRet;
}

bool nn_layer::set_all_node_values(float* pfValue)
{
    bool bRet = false;

    if(!m_bInitialized)
    {
        return bRet;
    }

    bRet = true;

    uint i = 0;

    for(i = 0; i < m_unNumNodes; i++)
    {
        m_ppNodes[i]->set_value(pfValue[i]);
        //NNLOG_MIL("i:%d node[i]->get_value():%f value[i]:%f", i, nodes[i]->get_value(), value[i]);
    }

    
    return bRet;
}

bool nn_layer::set_all_node_biases(float* pfBias)
{
    bool bRet = false;

    if(!m_bInitialized)
    {
        return bRet;
    }

    bRet = true;

    uint i = 0;

    for(i = 0; i < m_unNumNodes; i++)
    {
        m_ppNodes[i]->set_bias(pfBias[i]);
    }

    
    return bRet;

}

bool nn_layer::set_node_bias(float fBias, uint unIdx)
{
    bool bRet = false;

    if((unIdx < m_unNumNodes) && m_bInitialized)
    {
        m_ppNodes[unIdx]->set_bias(fBias);

        bRet = true;
    }

    
    return bRet;
}

bool nn_layer::set_node_delta(float fDelta, uint unIdx)
{
    bool bRet = false;

    if((unIdx < m_unNumNodes) && m_bInitialized)
    {
        m_ppNodes[unIdx]->set_delta(fDelta);

        bRet = true;
    }

    
    return bRet;
}

bool nn_layer::is_initialized()
{
    return m_bInitialized;
}

float nn_layer::get_node_value_idx(uint unIdx)
{
    return m_ppNodes[unIdx]->get_value();
}

float nn_layer::get_node_bias_idx(uint unIdx)
{
    return m_ppNodes[unIdx]->get_bias();
}

float nn_layer::get_node_delta_idx(uint unIdx)
{
    return m_ppNodes[unIdx]->get_delta();
}

void nn_layer::set_layer_type(eLyr_type eLType)
{
    eLyrType = eLType;
}

eLyr_type nn_layer::get_layer_type()
{
    return eLyrType;

}

void nn_layer::populateBiasesWithRandomNumbers()
{
    if(!m_bInitialized)
    {
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


}
