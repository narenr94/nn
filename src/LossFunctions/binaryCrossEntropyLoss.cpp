#include "binaryCrossEntropyLoss.h"
#include "nn_core.h"

BinaryCrossEntropyLoss::BinaryCrossEntropyLoss(NeuralNet* nn)
{
    m_pNN = nn;
    m_unOutputLyrID = (m_pNN->GetNumLys() - 1);
    m_unOutputLyrSz = m_pNN->GetSzLayer(m_unOutputLyrID);
}

float BinaryCrossEntropyLoss::apply_loss_func(float* fExpOut)
{
    float fRet = 0.0f;
    float temp = 0.0f;

    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        //ce += y_true[i] * std::log(y_pred[i]) + (1 - y_true[i]) * std::log(1 - y_pred[i]);
        temp = (fExpOut[i] * std::log(m_pNN->GetNodeVal(m_unOutputLyrID, i))) + ((1.0f - fExpOut[i]) * std::log(1.0f - m_pNN->GetNodeVal(m_unOutputLyrID, i)));
        fRet += temp;
    }

    return (-1.0f * (fRet/m_unOutputLyrSz));
}

void BinaryCrossEntropyLoss::get_loss_func_derv(float* fExpOut, float* fVal)
{

    // printf("\ndiff:%f\n", (fExpOut - m_pNN->GetNodeVal(m_unOutputLyrID, idx)));
    // fflush(stdout);
    float temp1;
    float temp2;
    float temp3;

    // printf("\n\n\n");
    // fflush(stdout);

    for(uint i = 0; i < m_unOutputLyrSz; i++)
    {
        // printf("\npredicted:%f expected:%f\n", m_pNN->GetNodeVal(m_unOutputLyrID, i), fExpOut[i]);
        // fflush(stdout);
        // temp1 = (m_pNN->GetNodeVal(m_unOutputLyrID, i) - fExpOut[i]);
        // temp2 = (1.0f - m_pNN->GetNodeVal(m_unOutputLyrID, i) );
        // temp3 = 
        fVal[i] = ((m_pNN->GetNodeVal(m_unOutputLyrID, i) - fExpOut[i]) / (m_pNN->GetNodeVal(m_unOutputLyrID, i) * (1.0f - m_pNN->GetNodeVal(m_unOutputLyrID, i) )));
        // printf("\nBCE loss derv:%f\n", fVal[i]);
        // fflush(stdout);
    }

}

BinaryCrossEntropyLoss::~BinaryCrossEntropyLoss(){}