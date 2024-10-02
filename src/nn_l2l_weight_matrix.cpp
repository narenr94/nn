#include "nn_l2l_weight_matrix.h"

nn_l2l_weight_matrix::nn_l2l_weight_matrix(nn_layer* pInL, nn_layer* pOutL)
{
    NNLOG_TRACE("entering");

    if(pInL->is_initialized() && pOutL->is_initialized())
    {
        m_pInputLayer = pInL;
        m_pOutputLayer = pOutL;

        uint in_l_size = m_pInputLayer->get_num_nodes();
        uint out_l_size = m_pOutputLayer->get_num_nodes();

        m_unSize = in_l_size * out_l_size;

        m_pfWeightMatrix = (float*)malloc(m_unSize * sizeof(float));

        m_bInitialized = true;

    }

    NNLOG_TRACE("exiting");
}


nn_l2l_weight_matrix::~nn_l2l_weight_matrix()
{
    NNLOG_TRACE("entering");
    if(m_pfWeightMatrix)
    {
        free(m_pfWeightMatrix);
    }
    NNLOG_TRACE("exiting");
}


bool nn_l2l_weight_matrix::set_weight(uint unInIdx, uint unOutIdx, float fWt)
{
    NNLOG_TRACE("entering in_idx:%d out_idx:%d wt:%f", unInIdx, unOutIdx, fWt);

    bool bRet = false;

    if(m_bInitialized)
    {
        if((unInIdx < m_pInputLayer->get_num_nodes()) && (unOutIdx < m_pOutputLayer->get_num_nodes()))
        {
            m_pfWeightMatrix[(unInIdx * m_pOutputLayer->get_num_nodes()) + unOutIdx] = fWt;

            bRet = true;
        }

    }
    else
    {
        NNLOG_ERR("exiting not initialized!!!");
    }

    NNLOG_TRACE("exiting");

    return bRet;

}

bool nn_l2l_weight_matrix::set_all_weight(float* fWt)
{
    NNLOG_TRACE("entering");
    bool bRet = false;

    if(m_bInitialized)
    {
        uint i = 0;
        uint j = 0;
        for(i = 0; i < m_pInputLayer->get_num_nodes(); i++)
        {
            for(j = 0; j < m_pOutputLayer->get_num_nodes(); j++)
            {
                m_pfWeightMatrix[(i * m_pOutputLayer->get_num_nodes()) + j] = fWt[(i * m_pOutputLayer->get_num_nodes()) + j];
            }
        }

        bRet = true;

    }
    else
    {
        NNLOG_ERR("exiting not initialized!!!");
    }

    NNLOG_TRACE("exiting");
    return bRet;
}

float nn_l2l_weight_matrix::get_weight(uint unInIdx, uint unOutIdx)
{
    NNLOG_TRACE("entering in_idx:%d out_idx:%d", unInIdx, unOutIdx);
    float bRet = 0;
    if(m_bInitialized)
    {
        bRet = m_pfWeightMatrix[(unInIdx * m_pOutputLayer->get_num_nodes()) + unOutIdx];
    }
    else
    {
        NNLOG_ERR("exiting not initialized!!!");
    }

    NNLOG_TRACE("exiting");

    return bRet;

}

uint nn_l2l_weight_matrix::get_size()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return m_unSize;
}

void nn_l2l_weight_matrix::populateWeightsWithRandomNumbers()
{
    NNLOG_TRACE("entering");
    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting not initialized!!!");
        return;
    }

    uint i = 0;

    for(i = 0; i < m_unSize; i++)
    {
        m_pfWeightMatrix[i] = (float)getRandomNumber(RAND_MIN_WEIGHT_BIAS, RAND_MAX_WEIGHT_BIAS);
        m_pfWeightMatrix[i] /= 10.0;
    }

    NNLOG_TRACE("exiting");
}

