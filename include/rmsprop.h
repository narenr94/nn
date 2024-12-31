#ifndef RMSPROP_H
#define RMSPROP_H

#include "optimizer.h"

struct RMSPropLayerRep{
    float* E_Val = nullptr;

    RMSPropLayerRep(uint size)
    {
        E_Val = new float[size];
        for(uint i = 0; i < size; i++)
        {
            E_Val[i] = 0.0f; 
        }
    }

    ~RMSPropLayerRep()
    {
        if(E_Val)
        {
            delete [] E_Val; 
        }
    }
};

struct RMSPropMtxRep{

    float* E_Val = nullptr;
    RMSPropMtxRep(uint size)
    {
        E_Val = new float[size];
        for(uint i = 0; i < size; i++)
        {
            E_Val[i] = 0.0f; 
        }
    }

    ~RMSPropMtxRep()
    {
        if(E_Val)
        {
            delete [] E_Val; 
        }
    }

};


class RMSProp : public Optimizer
{
    private:
        float m_fBeta;
        float m_fEpsilon;
        RMSPropLayerRep** m_ppLyrRep = nullptr;
        RMSPropMtxRep** m_ppMtxRep = nullptr;

        void correct_biases();

        void correct_weights();

    public:
    
        RMSProp(NeuralNet* nn, float beta = 0.9f, float epslion = 0.00000001f);

        void correct_weights_biases();

        ~RMSProp();
};


#endif