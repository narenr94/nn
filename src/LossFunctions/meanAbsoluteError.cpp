#include "meanAbsoluteError.h"
#include "nn_core.h"

MeanAbsoluteError::MeanAbsoluteError(NeuralNet* nn)
{
    m_pNN = nn;
    m_unOutputLyrID = (m_pNN->GetNumLys() - 1);
    m_unOutputLyrSz = m_pNN->GetSzLayer(m_unOutputLyrID);
}

float MeanAbsoluteError::apply_loss_func(float* fExpOut)
{
    float fRet = 0.0f;
    float temp = 0.0f;

    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        temp = std::fabs(fExpOut[i] - m_pNN->GetNodeVal(m_unOutputLyrID, i));
        fRet += temp;
    }

    return (fRet/m_unOutputLyrSz);
}

void MeanAbsoluteError::get_loss_func_derv(float* fExpOut, float* fVal)
{

    // printf("\ndiff:%f\n", (fExpOut - m_pNN->GetNodeVal(m_unOutputLyrID, idx)));
    // fflush(stdout);

    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        if (m_pNN->GetNodeVal(m_unOutputLyrID, i) > fExpOut[i]) 
        { 
            fVal[i] = 1.0f / m_unOutputLyrSz; 
            // fRet = 1.0f;
        } else if (m_pNN->GetNodeVal(m_unOutputLyrID, i) < fExpOut[i]) 
        { 
            fVal[i] = -1.0f / m_unOutputLyrSz; 
            // fRet = -1.0f;
        } else 
        { 
            fVal[i] = 0.0f; 
        }
    }

}

MeanAbsoluteError::~MeanAbsoluteError(){}