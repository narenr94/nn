#include "adam.h"
#include "nn_core.h"
#include <cmath>

ADAMOPT::ADAMOPT(NeuralNet* nn, float beta1, float beta2,float epslion)
{
    m_pNN = nn;
    m_fBeta1 = beta1;
    m_fBeta1 = beta2;
    m_fEpsilon = epslion;

    uint numLys = m_pNN->GetNumLys();

    m_ppLyrRep = new ADAMLayerRep*[numLys];
    m_ppMtxRep = new ADAMMtxRep*[numLys - 1];

    for(uint i = 0; i < numLys; i++)
    {
        m_ppLyrRep[i] = new ADAMLayerRep(m_pNN->GetSzLayer(i));
        if(i < (numLys - 1))
        {
            m_ppMtxRep[i] = new ADAMMtxRep(m_pNN->GetSzMtx(i + 1));
        }
    }

    m_unTimeStep = 0;
    
}

ADAMOPT::~ADAMOPT()
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

void ADAMOPT::correct_biases()
{
    float m_hat;
    float v_hat;
    uint i; //in layer index, out layer index is always in layer index + 1
    uint j; //in layer node index

    for(i = 1; i < m_pNN->GetNumLys(); i++)
    {
        for(j = 0; j < m_pNN->GetSzLayer(i); j++)
        {
            m_ppLyrRep[i]->M_Val[j] = m_fBeta1 * m_ppLyrRep[i]->M_Val[j] + (1.0f - m_fBeta1) * m_pNN->GetDelta(i, j);
            m_ppLyrRep[i]->V_Val[j] = m_fBeta2 * m_ppLyrRep[i]->V_Val[j] + (1.0f - m_fBeta2) * m_pNN->GetDelta(i, j) * m_pNN->GetDelta(i, j);
            m_hat = m_ppLyrRep[i]->M_Val[j] / (1.0f - std::pow(m_fBeta1, m_unTimeStep));
            v_hat = m_ppLyrRep[i]->V_Val[j] / (1.0f - std::pow(m_fBeta2, m_unTimeStep));

            m_pNN->SetBias(i, j, m_pNN->GetBias(i, j) - ((m_pNN->GetLearningRate() * m_hat)/(std::sqrt(v_hat) + m_fEpsilon)));
            
        }
    }
}

void ADAMOPT::correct_transform_parameters()
{

    double m_hat;
    double v_hat;
    uint i; //in layer index, out layer index is always in layer index + 1
    uint j; //in layer node index
    uint k; //out layer node index

    double delta_wt = 0.0;

    for(i = 1; i < m_pNN->GetNumLys(); i++)
    {
        for(j = 0; j < m_pNN->GetSzLayer(i - 1); j++)
        {
            for(k = 0; k < m_pNN->GetSzLayer(i); k++)
            {
                uint eval_idx = (j * m_pNN->GetSzLayer(i)) + k;

                delta_wt = m_pNN->GetDelta(i, k) * m_pNN->GetNodeVal(i - 1, j);
                m_ppMtxRep[i - 1]->M_Val[eval_idx] = (m_fBeta1 * m_ppMtxRep[i - 1]->M_Val[eval_idx]) + ((1.0f - m_fBeta1) * delta_wt);
                m_ppMtxRep[i - 1]->V_Val[eval_idx] = (m_fBeta2 * m_ppMtxRep[i - 1]->V_Val[eval_idx]) + ((1.0f - m_fBeta2) * delta_wt * delta_wt);
                m_hat = m_ppMtxRep[i - 1]->M_Val[eval_idx] / (1.0f - std::pow(m_fBeta1, m_unTimeStep));
                v_hat = m_ppMtxRep[i - 1]->V_Val[eval_idx] / (1.0f - std::pow(m_fBeta2, m_unTimeStep));
                uint mtx_idx = (j * m_pNN->GetSzLayer(i)) + k;
                m_pNN->SetWeight(i, j, k, (m_pNN->GetWeight(i, mtx_idx) - ((m_pNN->GetLearningRate() * m_hat)/(std::sqrt(v_hat) + m_fEpsilon))));
                
            }
            

        }
    }
}

void ADAMOPT::correct_transform_parameters_and_biases()
{
    m_unTimeStep++;
    correct_transform_parameters();
    correct_biases();   

}