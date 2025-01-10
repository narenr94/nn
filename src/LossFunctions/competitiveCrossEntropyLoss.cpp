#include "competitiveCrossEntropyLoss.h"
#include "nn_core.h"

CompetitiveCrossEntropyLoss::CompetitiveCrossEntropyLoss(NeuralNet* nn)
{
    m_pNN = nn;
    m_unOutputLyrID = (m_pNN->GetNumLys() - 1);
    m_unOutputLyrSz = m_pNN->GetSzLayer(m_unOutputLyrID);
}

float CompetitiveCrossEntropyLoss::apply_loss_func(float* fExpOut)
{
    float fRet = 0.0f;
    float temp = 0.0f;

    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        //loss += actual[i] * std::log(predicted[i]);
        temp = fExpOut[i] * std::log(m_pNN->GetNodeVal(m_unOutputLyrID, i));
        fRet += temp;
    }

    return (-1.0f * (fRet/m_unOutputLyrSz));

    
}

void CompetitiveCrossEntropyLoss::get_loss_func_derv(float* fExpOut, float* fVal)
{

    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        fVal[i] = ((-1.0f * fExpOut[i]) / (m_pNN->GetNodeVal(m_unOutputLyrID, i)));
    }

}

CompetitiveCrossEntropyLoss::~CompetitiveCrossEntropyLoss(){}