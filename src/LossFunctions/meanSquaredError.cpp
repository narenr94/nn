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

float MeanSquaredError::apply_loss_func_derv(float fExpOut, uint idx)
{

    // printf("\ndiff:%f\n", (fExpOut - m_pNN->GetNodeVal(m_unOutputLyrID, idx)));
    // fflush(stdout);

    return (-2.0f * ((fExpOut - m_pNN->GetNodeVal(m_unOutputLyrID, idx)) / m_unOutputLyrSz));
}

MeanSquaredError::~MeanSquaredError(){}