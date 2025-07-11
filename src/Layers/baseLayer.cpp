#include "baseLayer.h"
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

void validate_input_data_layers(eLayer_type t_layer_type, sLayer_Dimensions t_dims);

BaseLayer::BaseLayer(eLayer_type t_layer_type, sLayer_Dimensions t_dims, eAct_func eActFunc, float actParam1)
{
    m_layer_type = t_layer_type;

    validate_input_data_layers(t_layer_type, t_dims);
    
    m_Dimensions = t_dims;

    m_Dimensions.unInputRows = t_dims.unInputRows;
    m_Dimensions.unInputColumns = t_dims.unInputColumns;

    m_Dimensions.unOutputRows = t_dims.unOutputRows;
    m_Dimensions.unOutputColumns = t_dims.unOutputColumns;

    m_unNumNodes = m_Dimensions.unOutputRows * m_Dimensions.unOutputColumns;

    m_eActFunc = eActFunc;
    m_actParam1 = actParam1;

    //filter = (1 + in) - out
    m_Dimensions.unTransformParametersRows = t_dims.unTransformParametersRows;
    m_Dimensions.unTransformParametersColumns = t_dims.unTransformParametersColumns;

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

BaseLayer::~BaseLayer()
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


void BaseLayer::do_backwardpass_to_previous_layer()
{
    sLayer_Dimensions dims = m_pNextLyr->get_layer_dimensions();
    
    switch(m_pNextLyr->get_layer_type())
    {
        case eLayer_type::CONV :
            m_pAccelerator->do_backwardpass_conv_layer(dims);
            break;
        case eLayer_type::DENSE :
            m_pAccelerator->do_backwardpass_dense_layer();
        default :
            m_pAccelerator->do_backwardpass_dense_layer();
    }
}

uint BaseLayer::get_num_nodes()
{
    return m_unNumNodes;
    
}

bool BaseLayer::set_node_value(float fVal, uint unIdx)
{
    bool bRet = false;

    if(unIdx < m_unNumNodes)
    {
        m_pfValues[unIdx] = fVal;

        bRet = true;
    }

    
    return bRet;
}

bool BaseLayer::set_all_node_values(float* pfValue)
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

bool BaseLayer::set_all_node_biases(float* pfBias)
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

bool BaseLayer::set_node_bias(float fBias, uint unIdx)
{
    bool bRet = false;

    if((unIdx < m_unNumNodes))
    {
        m_pfBiases[unIdx] = fBias;

        bRet = true;
    }

    
    return bRet;
}

bool BaseLayer::set_node_delta(float fDelta, uint unIdx)
{
    bool bRet = false;

    if((unIdx < m_unNumNodes))
    {
        m_pfDeltas[unIdx] = fDelta;
        bRet = true;
    }

    
    return bRet;
}

float BaseLayer::get_node_value_idx(uint unIdx)
{
    return m_pfValues[unIdx];
}

float BaseLayer::get_node_bias_idx(uint unIdx)
{
    return m_pfBiases[unIdx];
}

float BaseLayer::get_node_delta_idx(uint unIdx)
{
    return m_pfDeltas[unIdx];
}

void BaseLayer::populateBiasesWithRandomNumbers()
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

eAct_func BaseLayer::get_act_func()
{
    return m_eActFunc;
}

float BaseLayer::get_act_param()
{
    return m_actParam1;
}

void BaseLayer::apply_act_func_all_nodes()
{
    
    m_pActFunc->apply_act_func();
    
}

void BaseLayer::get_delta_all_nodes(float * fVal)
{
    
    if(m_layer_type != eLayer_type::INPUT)
    {
        m_pActFunc->get_delta(fVal);
    }    
    
}

const float* BaseLayer::get_transform_matrix()
{
    assert(m_bPrevNxtLyrsSet == true);
    return m_pfTransformParameters;
}

BaseLayer* BaseLayer::GetPreviousLayer()
{
    return m_pPrevLyr;
}

BaseLayer* BaseLayer::GetNextLayer()
{
    return m_pNextLyr;
}

void BaseLayer::do_backwardpass_to_previous_layer_output_layer(float* fExpOut, BaseLossFunction* lossFunc)
{
    assert(m_bPrevNxtLyrsSet == true);
    m_pAccelerator->do_backwardpass_from_output_layer(fExpOut, lossFunc);
}

float BaseLayer::get_transform_matrix_parameter(uint Idx)
{
    assert(m_bPrevNxtLyrsSet == true);
    assert(Idx <= m_unTransformMatrixSize);
    float bRet = 0;
    
    bRet = m_pfTransformParameters[Idx];

    return bRet;

}

uint BaseLayer::get_transform_matrix_parameter_size()
{
    return m_unTransformMatrixSize;
}

void BaseLayer::populate_transform_matrix_parameter_with_random_numbers()
{
    assert(m_bPrevNxtLyrsSet == true);
    uint i = 0;

    for(i = 0; i < m_unTransformMatrixSize; i++)
    {
        m_pfTransformParameters[i] = (float)getRandomNumber(RAND_MIN_PARAMETER_BIAS, RAND_MAX_PARAMETER_BIAS);
        m_pfTransformParameters[i] /= 10.0;
    }
}

eLayer_type BaseLayer::get_layer_type()
{
    return m_layer_type;
}

sLayer_Dimensions BaseLayer::get_layer_dimensions()
{
    return m_Dimensions;
}

void validate_input_data_layers(eLayer_type t_layer_type, sLayer_Dimensions t_dims)
{

    

    if(t_layer_type == eLayer_type::INPUT)
    {
        assert(t_dims.unInputRows == 0);
        assert(t_dims.unInputColumns == 0);
        assert(t_dims.unNoTransformParameterMtx == 0);
        assert(t_dims.unTransformParametersColumns == 0);
        assert(t_dims.unTransformParametersRows == 0);
    }

    if(t_layer_type == eLayer_type::CONV)
    {
        assert((t_dims.unInputRows > 0) && (t_dims.unInputColumns > 0));

    }

    if(t_layer_type == eLayer_type::DENSE)
    {
        assert((t_dims.unInputRows > 0) && (t_dims.unInputColumns > 0));
        assert(t_dims.unNoTransformParameterMtx == 1);
        assert(((t_dims.unInputRows * t_dims.unInputColumns) * (t_dims.unOutputRows * t_dims.unOutputColumns)) == (t_dims.unTransformParametersColumns * t_dims.unTransformParametersRows));
    }

}
