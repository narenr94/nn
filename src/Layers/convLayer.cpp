#include "convLayer.h"

#include "nn_math.h"

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

#include <cstdio>

ConvLayer::ConvLayer(uint in_row, uint in_col, uint out_row, uint out_col, eAct_func eActFunc, float actParam1):BaseLayer((out_row * out_col), eActFunc, actParam1),
m_unKernelColumns(0),
m_unKernelRows(0)
{
    assert((in_row > 0) && (in_col > 0));

    assert((in_row * in_col) > (out_col * out_row));

    m_unOutputRows = out_row;
    m_unOutputColumns = out_col;

    m_unInputRows = in_row;
    m_unInputColumns = in_col;

    m_unNumNodes = out_col * out_row;

    m_eActFunc = eActFunc;
    m_actParam1 = actParam1;

    //filter = (1 + in) - out
    m_unKernelRows = (1 + m_unInputRows) - m_unOutputRows;
    m_unKernelColumns = (1 + m_unInputColumns) - m_unOutputColumns;

    m_pfValues = new float[m_unNumNodes];
    m_pfBiases = new float[m_unNumNodes];
    m_pfDeltas = new float[m_unNumNodes];

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

ConvLayer::~ConvLayer()
{
    if(m_pfValues)
    {
        delete [] m_pfValues;
    }

    if(m_pfBiases)
    {
        delete [] m_pfBiases;
    }

    if(m_pfDeltas)
    {
        delete [] m_pfDeltas;
    }

    if(m_pActFunc)
    {
        delete m_pActFunc;
    }

    if(m_pfTransformParameters)
    {
        delete [] m_pfTransformParameters;
    }
}

uint ConvLayer::get_num_nodes()
{
    return m_unNumNodes;
    
}

bool ConvLayer::set_node_value(float fVal, uint unIdx)
{
    bool bRet = false;

    if(unIdx < m_unNumNodes)
    {
        m_pfValues[unIdx] = fVal;

        bRet = true;
    }

    
    return bRet;
}

bool ConvLayer::set_all_node_values(float* pfValue)
{
    bool bRet = false;

    bRet = true;

    uint i = 0;

    for(i = 0; i < m_unNumNodes; i++)
    {
        m_pfValues[i] = pfValue[i];
        //NNLOG_MIL("i:%d node[i]->get_value():%f value[i]:%f", i, nodes[i]->get_value(), value[i]);
    }

    
    return bRet;
}

bool ConvLayer::set_all_node_biases(float* pfBias)
{
    bool bRet = false;

    bRet = true;

    uint i = 0;

    for(i = 0; i < m_unNumNodes; i++)
    {
        m_pfBiases[i] = pfBias[i];
    }

    
    return bRet;

}

bool ConvLayer::set_node_bias(float fBias, uint unIdx)
{
    bool bRet = false;

    if((unIdx < m_unNumNodes))
    {
        m_pfBiases[unIdx] = fBias;

        bRet = true;
    }

    
    return bRet;
}

bool ConvLayer::set_node_delta(float fDelta, uint unIdx)
{
    bool bRet = false;

    if((unIdx < m_unNumNodes))
    {
        m_pfDeltas[unIdx] = fDelta;
        bRet = true;
    }

    
    return bRet;
}

float ConvLayer::get_node_value_idx(uint unIdx)
{
    return m_pfValues[unIdx];
}

float ConvLayer::get_node_bias_idx(uint unIdx)
{
    return m_pfBiases[unIdx];
}

float ConvLayer::get_node_delta_idx(uint unIdx)
{
    return m_pfDeltas[unIdx];
}

void ConvLayer::populateBiasesWithRandomNumbers()
{
    uint i = 0;
    float tmp;

    for(i = 0; i < m_unNumNodes; i++)
    {
        tmp = (float)getRandomNumber(RAND_MIN_PARAMETER_BIAS, RAND_MAX_PARAMETER_BIAS);
        tmp /= 10.0;
        m_pfBiases[i] = tmp;
    }


}

eAct_func ConvLayer::get_act_func()
{
    return m_eActFunc;
}

float ConvLayer::get_act_param()
{
    return m_actParam1;
}

void ConvLayer::apply_act_func_all_nodes()
{
    
    m_pActFunc->apply_act_func();
    
}

void ConvLayer::get_delta_all_nodes(float * fVal)
{
    
    m_pActFunc->get_delta(fVal);
    
}

const float* ConvLayer::get_transform_matrix()
{
    assert(m_bPrevNxtLyrsSet == true);
    return m_pfTransformParameters;
}

void ConvLayer::SetPreviousNextLayers(BaseLayer* prevLyr, BaseLayer* nxtLyr)
{
    m_pPrevLyr = prevLyr;
    m_pNextLyr = nxtLyr;

    m_unTransformMatrixSize = 0;

    if(m_pPrevLyr)
    {
        assert(m_pPrevLyr->get_num_nodes() == (m_unInputRows * m_unInputColumns));
        m_unTransformMatrixSize = m_unKernelRows * m_unKernelColumns;
        m_pfTransformParameters = new float[m_unTransformMatrixSize];
    }

    m_bPrevNxtLyrsSet = true;
}

BaseLayer* ConvLayer::GetPreviousLayer()
{
    return m_pPrevLyr;
}

BaseLayer* ConvLayer::GetNextLayer()
{
    return m_pNextLyr;
}

void ConvLayer::do_forwardpass_to_current_layer()
{
    assert(m_bPrevNxtLyrsSet == true);
    m_pAccelerator->do_forwardpass_conv_layer(m_unInputRows, m_unInputColumns, m_unKernelRows, m_unKernelColumns);
}

void ConvLayer::do_backwardpass_to_previous_layer_output_layer(float* fExpOut, BaseLossFunction* lossFunc)
{
    assert(m_bPrevNxtLyrsSet == true);
    // m_pAccelerator->do_backwardpass_conv_layer_output_layer(fExpOut, lossFunc);
}

void ConvLayer::do_backwardpass_to_previous_layer()
{
    assert(m_bPrevNxtLyrsSet == true);
    // m_pAccelerator->do_backwardpass_conv_layer();
}

void ConvLayer::set_transform_matrix_parameter(uint unInIdx, uint unOutIdx, float fWt)
{
    assert(m_bPrevNxtLyrsSet == true);
    assert((unInIdx < m_pPrevLyr->get_num_nodes()) && (unOutIdx < get_num_nodes()));

    m_pfTransformParameters[(unInIdx * get_num_nodes()) + unOutIdx] = fWt;

}

void ConvLayer::set_transform_matrix_parameter(uint Idx, float fWt)
{
    assert(m_bPrevNxtLyrsSet == true);
    assert(Idx < m_unTransformMatrixSize);
    
    m_pfTransformParameters[Idx] = fWt;

}



void ConvLayer::set_all_transform_matrix_parameter(float* fWt)
{
    assert(m_bPrevNxtLyrsSet == true);
    uint i = 0;
    uint j = 0;
    for(i = 0; i < m_unKernelRows; i++)
    {
        for(j = 0; j < m_unKernelColumns; j++)
        {
            m_pfTransformParameters[(i * m_unKernelColumns) + j] = fWt[(i * m_unKernelColumns) + j];
        }
    }
}

float ConvLayer::get_transform_matrix_parameter(uint Idx)
{
    assert(m_bPrevNxtLyrsSet == true);
    assert(Idx <= m_unTransformMatrixSize);
    float bRet = 0;
    
    bRet = m_pfTransformParameters[Idx];

    return bRet;

}

uint ConvLayer::get_transform_matrix_parameter_size()
{
    return m_unTransformMatrixSize;
}

void ConvLayer::populate_transform_matrix_parameter_with_random_numbers()
{
    assert(m_bPrevNxtLyrsSet == true);
    uint i = 0;

    for(i = 0; i < m_unTransformMatrixSize; i++)
    {
        m_pfTransformParameters[i] = (float)getRandomNumber(RAND_MIN_PARAMETER_BIAS, RAND_MAX_PARAMETER_BIAS);
        m_pfTransformParameters[i] /= 10.0;
    }
}


