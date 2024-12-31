#ifndef ADAM_H
#define ADAM_H

#include "optimizer.h"

struct ADAMLayerRep{
    float* M_Val = nullptr;
    float* V_Val = nullptr;

    ADAMLayerRep(uint size)
    {
        M_Val = new float[size];
        V_Val = new float[size];
        for(uint i = 0; i < size; i++)
        {
            M_Val[i] = 0.0f;
            V_Val[i] = 0.0f; 
        }
    }

    ~ADAMLayerRep()
    {
        if(M_Val)
        {
            delete [] M_Val; 
        }
        if(V_Val)
        {
            delete [] V_Val; 
        }
    }
};

struct ADAMMtxRep{

    float* M_Val = nullptr;
    float* V_Val = nullptr;

    ADAMMtxRep(uint size)
    {
        M_Val = new float[size];
        V_Val = new float[size];
        for(uint i = 0; i < size; i++)
        {
            M_Val[i] = 0.0f;
            V_Val[i] = 0.0f; 
        }
    }

    ~ADAMMtxRep()
    {
        if(M_Val)
        {
            delete [] M_Val; 
        }
        if(V_Val)
        {
            delete [] V_Val; 
        }
    }

};


class ADAMOPT : public Optimizer
{
    private:
        float m_fBeta1;
        float m_fBeta2;
        float m_fEpsilon;
        ADAMLayerRep** m_ppLyrRep = nullptr;
        ADAMMtxRep** m_ppMtxRep = nullptr;
        uint m_unTimeStep = 0;

        void correct_biases();

        void correct_weights();

    public:
        //KLUDGE : default beta1:0.9 beta2:0.999 epsilon0.00000001, currently set to work for mnist example
        ADAMOPT(NeuralNet* nn, float beta1 = 0.9f, float beta2 = 0.9f,float epslion = 0.1f);
        
        void correct_weights_biases();

        ~ADAMOPT();
};


#endif