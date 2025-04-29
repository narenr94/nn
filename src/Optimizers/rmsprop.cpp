#include "rmsprop.h"
#include "nn_core.h"
#include <cmath>

RMSProp::RMSProp(NeuralNet* nn, float beta, float epslion)
{
    m_pNN = nn;
    m_fBeta = beta;
    m_fEpsilon = epslion;

    uint numLys = m_pNN->GetNumLys();

    m_ppLyrRep = new RMSPropLayerRep*[numLys];
    m_ppMtxRep = new RMSPropMtxRep*[numLys - 1];

    for(uint i = 0; i < numLys; i++)
    {
        m_ppLyrRep[i] = new RMSPropLayerRep(m_pNN->GetSzLayer(i));
        if(i < (numLys - 1))
        {
            m_ppMtxRep[i] = new RMSPropMtxRep(m_pNN->GetSzMtx(i + 1));
        }
    }
    
}

RMSProp::~RMSProp()
{

    if(m_ppLyrRep)
    {
        uint numLys = m_pNN->GetNumLys();

        for(uint i = 0; i < numLys; i++)
        {
            delete m_ppLyrRep[i];

            if(i < (numLys - 1))
            {
                delete m_ppMtxRep[i];
            }
        }

        delete [] m_ppLyrRep;
        delete [] m_ppMtxRep;
    }
}

void RMSProp::correct_biases()
{
    uint i; //in layer index, out layer index is always in layer index + 1
    uint j; //in layer node index

    for(i = 1; i < m_pNN->GetNumLys(); i++)
    {
        for(j = 0; j < m_pNN->GetSzLayer(i); j++)
        {
            m_ppLyrRep[i]->E_Val[j] = m_fBeta * m_ppLyrRep[i]->E_Val[j] + (1.0f - m_fBeta) * m_pNN->GetDelta(i, j) * m_pNN->GetDelta(i, j);
            m_pNN->SetBias(i, j, m_pNN->GetBias(i, j) - ((m_pNN->GetLearningRate() * m_pNN->GetDelta(i, j))/(std::sqrt(m_ppLyrRep[i]->E_Val[j]) + m_fEpsilon)));
            
        }
    }
}

void RMSProp::correct_weights()
{

    uint i; //in layer index, out layer index is always in layer index + 1
    uint j; //in layer node index
    uint k; //out layer node index

    float delta_wt = 0.0;

    for(i = 1; i < m_pNN->GetNumLys(); i++)
    {
        for(j = 0; j < m_pNN->GetSzLayer(i - 1); j++)
        {
            for(k = 0; k < m_pNN->GetSzLayer(i); k++)
            {
                uint eval_idx = (j * m_pNN->GetSzLayer(i)) + k;
                
                delta_wt = m_pNN->GetDelta(i, k) * m_pNN->GetNodeVal(i - 1, j);

                m_ppMtxRep[i - 1]->E_Val[eval_idx] = m_fBeta * m_ppMtxRep[i - 1]->E_Val[eval_idx] + (1.0f - m_fBeta) * delta_wt * delta_wt;

                delta_wt *= m_pNN->GetLearningRate();

                m_pNN->SetWeight(i, j, k, (m_pNN->GetWeight(i, j, k) - (delta_wt/(std::sqrt(m_ppMtxRep[i - 1]->E_Val[eval_idx]) + m_fEpsilon))));
            }
            

        }
    }
}

void RMSProp::correct_weights_biases()
{
    correct_weights();
    correct_biases();
}
