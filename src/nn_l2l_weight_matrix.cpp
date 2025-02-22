#include "nn_l2l_weight_matrix.h"

nn_l2l_weight_matrix::nn_l2l_weight_matrix(nn_layer* pInL, nn_layer* pOutL)
{
    if(pInL->is_initialized() && pOutL->is_initialized())
    {
        m_pInputLayer = pInL;
        m_pOutputLayer = pOutL;

        uint in_l_size = m_pInputLayer->get_num_nodes();
        uint out_l_size = m_pOutputLayer->get_num_nodes();

        m_unSize = in_l_size * out_l_size;

        m_pfWeightMatrix = new float [m_unSize];

        m_bInitialized = true;

    }

    
}


nn_l2l_weight_matrix::~nn_l2l_weight_matrix()
{
    if(m_pfWeightMatrix)
    {
        delete [] m_pfWeightMatrix;
        m_pfWeightMatrix = nullptr;
    }
    
}


bool nn_l2l_weight_matrix::set_weight(uint unInIdx, uint unOutIdx, float fWt)
{
    bool bRet = false;

    if(m_bInitialized)
    {
        if((unInIdx < m_pInputLayer->get_num_nodes()) && (unOutIdx < m_pOutputLayer->get_num_nodes()))
        {
            m_pfWeightMatrix[(unInIdx * m_pOutputLayer->get_num_nodes()) + unOutIdx] = fWt;

            bRet = true;
        }

    }
    

    return bRet;

}

bool nn_l2l_weight_matrix::set_weight(uint Idx, float fWt)
{
    bool bRet = false;

    if(m_bInitialized)
    {
        if(Idx < m_unSize)
        {
            m_pfWeightMatrix[Idx] = fWt;

            bRet = true;
        }

    }
    
    return bRet;

}



bool nn_l2l_weight_matrix::set_all_weight(float* fWt)
{
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
    
    return bRet;
}

float nn_l2l_weight_matrix::get_weight(uint unInIdx, uint unOutIdx)
{
    float bRet = 0;
    if(m_bInitialized)
    {
        bRet = m_pfWeightMatrix[(unInIdx * m_pOutputLayer->get_num_nodes()) + unOutIdx];
    }
    

    return bRet;

}

float nn_l2l_weight_matrix::get_weight(uint Idx)
{
    float bRet = 0;
    if(m_bInitialized)
    {
        if(Idx < m_unSize)
        {
            bRet = m_pfWeightMatrix[Idx];
        }
        
    }

    return bRet;

}

uint nn_l2l_weight_matrix::get_size()
{
    return m_unSize;
}

void nn_l2l_weight_matrix::populateWeightsWithRandomNumbers()
{
    if(!m_bInitialized)
    {
        return;
    }

    uint i = 0;

    for(i = 0; i < m_unSize; i++)
    {
        m_pfWeightMatrix[i] = (float)getRandomNumber(RAND_MIN_WEIGHT_BIAS, RAND_MAX_WEIGHT_BIAS);
        m_pfWeightMatrix[i] /= 10.0;
    }
}

const float* nn_l2l_weight_matrix::getWtMtx()
{
    return m_pfWeightMatrix;
}

