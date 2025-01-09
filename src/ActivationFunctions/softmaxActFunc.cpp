#include "softmaxActFunc.h"
#include "nn_layer.h"
#include <cstdio>

SoftmaxActFunc::SoftmaxActFunc(nn_layer* pNN_Layer)
{
    m_pNN_Layer = pNN_Layer;
    m_unSzLyr = m_pNN_Layer->get_num_nodes();
    m_ppDervMatrix = new float*[m_unSzLyr];
    for(uint i = 0; i < m_unSzLyr; i++)
    {
        m_ppDervMatrix[i] = new float[m_unSzLyr];
    }
}

SoftmaxActFunc::~SoftmaxActFunc()
{
    if(m_ppDervMatrix)
    {
        for(uint i = 0; i < m_unSzLyr; i++)
        {
            delete [] m_ppDervMatrix[i];
        }

        delete [] m_ppDervMatrix;
    }

}

void SoftmaxActFunc::apply_act_func()
{
    
    float max = m_pNN_Layer->get_node_value_idx(0);
    float sum = 0.0f;

    
    for(uint i = 1; i < m_unSzLyr; i++)
    {
        if(max < m_pNN_Layer->get_node_value_idx(i))
        {
            max = m_pNN_Layer->get_node_value_idx(i);
        }
    }
    for(uint i = 0; i < m_unSzLyr; i++)
    {
        float exp_val = get_softmaxf(m_pNN_Layer->get_node_value_idx(i), max);
        m_pNN_Layer->set_node_value(exp_val, i);
        sum += exp_val;
    }
    
    for(uint i = 0; i < m_unSzLyr; i++)
    {
        m_pNN_Layer->set_node_value(m_pNN_Layer->get_node_value_idx(i) / sum, i);
    }
    
}

void SoftmaxActFunc::populate_derv_matrix()
{
    for(uint i = 0; i < m_unSzLyr; i++)
    {
        for(uint j = 0; j < m_unSzLyr; j++)
        {
            if(i == j)
            {
                m_ppDervMatrix[i][j] = find_derivative_softmaxf(m_pNN_Layer->get_node_value_idx(i));
            }
            else
            {
                m_ppDervMatrix[i][j] = find_derivative_softmaxf_wrong_pred(m_pNN_Layer->get_node_value_idx(j), m_pNN_Layer->get_node_value_idx(i));
            }            

        }

    }
}

void SoftmaxActFunc::get_delta(float* fVal)
{

    uint i = 0;
    uint j = 0;

    populate_derv_matrix();

    float * fTemp = new float[m_unSzLyr];

    for(i = 0; i < m_unSzLyr; i++)
    {
        fTemp[i] = 0.0f;
    }
    
    for(i = 0; i < m_unSzLyr; i++)
    {
        for(j = 0; j < m_unSzLyr; j++)
        {
            fTemp[i] += fVal[j] * m_ppDervMatrix[i][j];
        }
        
    }
    
    for(i = 0; i < m_unSzLyr; i++)
    {
        fVal[i] = fTemp[i];
    }

    

    delete [] fTemp;
}
