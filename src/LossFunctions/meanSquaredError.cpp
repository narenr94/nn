#include "meanSquaredError.h"
#include "nn_core.h"
#include <cstdio>

MeanSquaredError::MeanSquaredError(NeuralNet* nn)
{
    m_pNN = nn;
    m_unOutputLyrID = (m_pNN->GetNumLys() - 1);
    m_unOutputLyrSz = m_pNN->GetSzLayer(m_unOutputLyrID);
}

float MeanSquaredError::apply_loss_func(float* fExpOut)
{
    float fRet = 0.0f;
    float temp = 0.0f;

    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        temp = fExpOut[i] - m_pNN->GetNodeVal(m_unOutputLyrID, i);
        temp *= temp;
        fRet += temp;
    }

    return (fRet/m_unOutputLyrSz);
}

void MeanSquaredError::get_loss_func_derv(float* fExpOut, float* fVal)
{

    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        fVal[i] = (-2.0f * ((fExpOut[i] - m_pNN->GetNodeVal(m_unOutputLyrID, i)) / m_unOutputLyrSz));
    }

}

MeanSquaredError::~MeanSquaredError(){}