#ifndef RMSPROP_H
#define RMSPROP_H

#include "baseOptimizer.h"

#define RMS_PROP_DEFAULT_BETA 0.9f
#define RMS_PROP_DEFAULT_EPSILON 0.00000001f

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


class RMSProp : public BaseOptimizer
{
    private:
        float m_fBeta;
        float m_fEpsilon;
        RMSPropLayerRep** m_ppLyrRep = nullptr;
        RMSPropMtxRep** m_ppMtxRep = nullptr;

        void correct_biases();

        void correct_transform_parameters();

    public:
    
        RMSProp(NeuralNet* nn, float beta = RMS_PROP_DEFAULT_BETA, float epslion = RMS_PROP_DEFAULT_EPSILON);

        void correct_transform_parameters_and_biases();

        ~RMSProp();

    private:

        void correct_transform_parameters_dense(uint curr_lyr_idx) override;

        void correct_transform_parameters_conv(uint curr_lyr_idx) override;
};


#endif