#include "nn_layer.h"

#include "nn_l2l_weight_matrix.h"

#include "sigmoidActFunc.h"
#include "reluActFunc.h"
#include "leakyReluActFunc.h"
#include "tanhActFunc.h"
#include "softmaxActFunc.h"

//Accelerators
#include "cpuAccelerator.h"
#ifdef OPENCL_ACC
#include "openclAccelerator.h"
#endif

#include <cassert>

uint nn_layer::get_num_nodes()
{
    return m_unNumNodes;
    
}


nn_layer::~nn_layer()
{
    if(m_ppNodes)
    {
        for(uint i = 0; i < m_unNumNodes; i++)
        {
            delete m_ppNodes[i];
        }
        delete [] m_ppNodes;
    }

    if(m_pActFunc)
    {
        delete m_pActFunc;
    }

    if(m_pPrevLyr)
    {
        delete m_pPrevLyr;
    }
}

nn_layer::nn_layer(uint unNumNodesnodes, eAct_func eActFunc, float actParam1)
:m_pNextLyr(nullptr),
m_pPrevLyr(nullptr),
m_pWtMtx(nullptr),
m_bPrevNxtLyrsSet(false)
{
    m_unNumNodes = unNumNodesnodes;

    m_eActFunc = eActFunc;

    m_actParam1 = actParam1;

    m_ppNodes = new nn_node*[m_unNumNodes];

    uint i = 0;

    for(i = 0; i < m_unNumNodes; i++)
    {
        m_ppNodes[i] = new nn_node();
    }

    switch(m_eActFunc)
    {
        case eAct_func::SIGMOID:
            m_pActFunc = new SigmoidActFunc(this);
            break;
        case eAct_func::RELU:
            m_pActFunc = new ReluActFunc(this);
            break;
        case eAct_func::LEAKY_RELU:
            m_pActFunc = new LeakyReluActFunc(this, m_actParam1 != 0.0f ? m_actParam1 : LEAKY_RELU_DEFAULT_ALPHA);
            break;
        case eAct_func::SOFTMAX:
            m_pActFunc = new SoftmaxActFunc(this);
            break;
        case eAct_func::TANH:
            m_pActFunc = new TanhActFunc(this);
            break;
        default:
            m_pActFunc = new SigmoidActFunc(this);
    }

#ifdef OPENCL_ACC
    m_pAccelerator = new OpenclAccelerator(this);
#else
    m_pAccelerator = new CpuAccelerator(this);
#endif

}

bool nn_layer::set_node_value(float fVal, uint unIdx)
{
    bool bRet = false;

    if(unIdx < m_unNumNodes)
    {
        m_ppNodes[unIdx]->set_value(fVal);

        bRet = true;
    }

    
    return bRet;
}

bool nn_layer::set_all_node_values(float* pfValue)
{
    bool bRet = false;

    bRet = true;

    uint i = 0;

    for(i = 0; i < m_unNumNodes; i++)
    {
        m_ppNodes[i]->set_value(pfValue[i]);
        //NNLOG_MIL("i:%d node[i]->get_value():%f value[i]:%f", i, nodes[i]->get_value(), value[i]);
    }

    
    return bRet;
}

bool nn_layer::set_all_node_biases(float* pfBias)
{
    bool bRet = false;

    bRet = true;

    uint i = 0;

    for(i = 0; i < m_unNumNodes; i++)
    {
        m_ppNodes[i]->set_bias(pfBias[i]);
    }

    
    return bRet;

}

bool nn_layer::set_node_bias(float fBias, uint unIdx)
{
    bool bRet = false;

    if((unIdx < m_unNumNodes))
    {
        m_ppNodes[unIdx]->set_bias(fBias);

        bRet = true;
    }

    
    return bRet;
}

bool nn_layer::set_node_delta(float fDelta, uint unIdx)
{
    bool bRet = false;

    if((unIdx < m_unNumNodes))
    {
        m_ppNodes[unIdx]->set_delta(fDelta);

        bRet = true;
    }

    
    return bRet;
}

float nn_layer::get_node_value_idx(uint unIdx)
{
    return m_ppNodes[unIdx]->get_value();
}

float nn_layer::get_node_bias_idx(uint unIdx)
{
    return m_ppNodes[unIdx]->get_bias();
}

float nn_layer::get_node_delta_idx(uint unIdx)
{
    return m_ppNodes[unIdx]->get_delta();
}

void nn_layer::set_layer_type(eLyr_type eLType)
{
    eLyrType = eLType;
}

eLyr_type nn_layer::get_layer_type()
{
    return eLyrType;

}

void nn_layer::populateBiasesWithRandomNumbers()
{
    uint i = 0;
    float tmp;

    for(i = 0; i < m_unNumNodes; i++)
    {
        tmp = (float)getRandomNumber(RAND_MIN_WEIGHT_BIAS, RAND_MAX_WEIGHT_BIAS);
        tmp /= 10.0;
        m_ppNodes[i]->set_bias(tmp);
    }


}

eAct_func nn_layer::get_act_func()
{
    return m_eActFunc;
}

float nn_layer::get_act_param()
{
    return m_actParam1;
}

void nn_layer::apply_act_func_all_nodes()
{
    
    m_pActFunc->apply_act_func();

    
}

void nn_layer::get_delta_all_nodes(float * fVal)
{
    
    m_pActFunc->get_delta(fVal);

    
}

nn_l2l_weight_matrix* nn_layer::GetWeightMatrix()
{
    return m_pWtMtx;
}

void nn_layer::SetPreviousNextLayers(nn_layer* prevLyr, nn_layer* nxtLyr)
{
    m_pPrevLyr = prevLyr;
    m_pNextLyr = nxtLyr;

    if(m_pPrevLyr)
    {
        m_pWtMtx = new nn_l2l_weight_matrix(m_pPrevLyr, this);
    }

    m_bPrevNxtLyrsSet = true;
}

nn_layer* nn_layer::GetPreviousLayer()
{
    return m_pPrevLyr;
}

nn_layer* nn_layer::GetNextLayer()
{
    return m_pNextLyr;
}

void nn_layer::do_forwardpass_to_current_layer()
{
    assert(m_bPrevNxtLyrsSet == true);
    m_pAccelerator->do_forwardpass_dense_layer();
}

void nn_layer::do_backwardpass_to_previous_layer_output_layer(float* fExpOut, BaseLossFunction* lossFunc)
{
    assert(m_bPrevNxtLyrsSet == true);
    m_pAccelerator->do_backwardpass_dense_layer_output_layer(fExpOut, lossFunc);
}

void nn_layer::do_backwardpass_to_previous_layer()
{
    assert(m_bPrevNxtLyrsSet == true);
    m_pAccelerator->do_backwardpass_dense_layer();
}

