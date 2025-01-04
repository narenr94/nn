#include "huberLoss.h"
#include "nn_core.h"

HuberLoss::HuberLoss(NeuralNet* nn, float delta)
{

    m_fDelta = delta;
    m_pNN = nn;
    m_unOutputLyrID = (m_pNN->GetNumLys() - 1);
    m_unOutputLyrSz = m_pNN->GetSzLayer(m_unOutputLyrID);
}

float HuberLoss::apply_loss_func(float* fExpOut)
{
    float fRet = 0.0f;
    float temp = 0.0f;

    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        // double diff = y_true[i] - y_pred[i]; 
        // if (std::abs(diff) <= delta) 
        // { 
        //     loss += 0.5 * diff * diff; 
        // } 
        // else 
        // { 
        //     loss += delta * (std::abs(diff) - 0.5 * delta); 
        // }
        float diff = fExpOut[i] - m_pNN->GetNodeVal(m_unOutputLyrID, i);
        if(std::abs(diff) <= m_fDelta)
        {
            temp = 0.5f * diff * diff;
        }
        else
        {
            temp = m_fDelta * (std::abs(diff) - (0.5f * m_fDelta));
        }
        fRet += temp;
    }

    return (fRet/m_unOutputLyrSz);
}

float HuberLoss::apply_loss_func_derv(float fExpOut, uint idx)
{
    float ret = 0.0f;
    // double diff = y_true[i] - y_pred[i]; 
    // if (std::abs(diff) <= delta) 
    // { 
    //     grad[i] = diff; 
    // } 
    // else 
    // { 
    //     grad[i] = delta * (diff < 0 ? -1 : 1); 
    // }
    double diff = fExpOut - m_pNN->GetNodeVal(m_unOutputLyrID, idx);
    if(std::abs(diff) <= m_fDelta)
    {
        // ret = diff;
        ret = -1.0f * diff;
    }
    else
    {
        // ret = m_fDelta * (diff < 0.0f ? -1.0f : 1.0f);
        ret = m_fDelta * (diff < 0.0f ? 1.0f : -1.0f);
    }

    return ret;
}

HuberLoss::~HuberLoss(){}