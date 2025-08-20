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

std::map<eAct_func, std::string> actFuncToString = {
        {SIGMOID, "SIGMOID"},
        {RELU, "RELU"},
        {LEAKY_RELU, "LEAKY_RELU"},
        {SOFTMAX, "SOFTMAX"},
        {TANH, "TANH"}
    };

std::map<std::string, eAct_func> stringToActFunc = {
        {"SIGMOID", SIGMOID},
        {"RELU", RELU},
        {"LEAKY_RELU", LEAKY_RELU},
        {"SOFTMAX", SOFTMAX},
        {"TANH", TANH}
    };

void validate_input_data_layers(eLayer_type t_layer_type, sLayer_Dimensions t_dims);

std::vector<std::string> split_by_lines(const std::string& block);

std::pair<std::string, std::vector<std::string>> parse_line(const std::string& line);

BaseLayer::BaseLayer(eLayer_type t_layer_type, sLayer_Dimensions t_dims, eAct_func eActFunc, float actParam1)
{
    setup_layer(t_layer_type, t_dims, eActFunc, actParam1);
}

BaseLayer::BaseLayer(eLayer_type t_layer_type, std::string load_data)
{
    m_layer_type = t_layer_type;

    std::vector<std::string> all_lines = split_by_lines(load_data);

    std::vector<std::pair<std::string, std::vector<std::string>>> parsed_lines;

    for(uint i = 0; i < all_lines.size(); i++)
    {
        parsed_lines.push_back(parse_line(all_lines[i]));
    }

    apply_parsed_load_data(t_layer_type, parsed_lines);

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
        case eLayer_type::POOLING :
            m_pAccelerator->do_backwardpass_pooling_layer(static_cast<ePooling_type>(static_cast<int>(m_pNextLyr->get_act_param())));
            break;
        case eLayer_type::DENSE :
            m_pAccelerator->do_backwardpass_dense_layer();
            break;
        default :
            assert(0); //unkown layer type
            break;
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

#ifndef IGNORE_LAYER_INPUT_VALIDATION_FOR_TESTING

    if(t_layer_type == eLayer_type::INPUT)
    {
        assert(t_dims.unInputRows == 0);
        assert(t_dims.unInputColumns == 0);
        assert(t_dims.unNoTransformParameterMtx == 1);
        assert(t_dims.unTransformParametersColumns == 0);
        assert(t_dims.unTransformParametersRows == 0);
    }

    if(t_layer_type == eLayer_type::CONV)
    {
        assert((t_dims.unInputRows > 0) && (t_dims.unInputColumns > 0));
        assert(t_dims.unNoTransformParameterMtx > 0);
    }

    if(t_layer_type == eLayer_type::DENSE)
    {
        assert((t_dims.unInputRows > 0) && (t_dims.unInputColumns > 0));
        assert(t_dims.unNoTransformParameterMtx == 1);
        assert(((t_dims.unInputRows * t_dims.unInputColumns) * (t_dims.unOutputRows * t_dims.unOutputColumns)) == (t_dims.unTransformParametersColumns * t_dims.unTransformParametersRows));
    }

    if(t_layer_type == eLayer_type::POOLING)
    {
        assert((t_dims.unInputRows > 0) && (t_dims.unInputColumns > 0));
        assert(t_dims.unNoTransformParameterMtx > 0);
    }

#endif

}


std::string BaseLayer::get_serialized_general_layer_data()
{
    std::ostringstream ss;
    ss << "dims: ";
    ss << m_Dimensions.unInputColumns << " ";
    ss << m_Dimensions.unInputRows << " ";
    ss << m_Dimensions.unNoTransformParameterMtx << " ";
    ss << m_Dimensions.unOutputColumns << " ";
    ss << m_Dimensions.unOutputRows << " ";
    ss << m_Dimensions.unTransformParametersColumns << " ";
    ss << m_Dimensions.unTransformParametersRows << " ";
    ss << "\n";
    ss << "act_func: ";
    assert(actFuncToString.find(m_eActFunc) != actFuncToString.end());
    ss << actFuncToString[m_eActFunc] << " \n";
    ss << "act_param: ";
    ss << m_actParam1 << " \n";

    return ss.str();
}

sLayer_Dimensions extract_dims(std::vector<std::string>& dims_string)
{
    assert(dims_string.size() == DIMS_SIZE);

    sLayer_Dimensions r_dims;

    r_dims.unInputColumns = std::stoul(dims_string[0]);
    r_dims.unInputRows = std::stoul(dims_string[1]);
    r_dims.unNoTransformParameterMtx = std::stoul(dims_string[2]);
    r_dims.unOutputColumns = std::stoul(dims_string[3]);
    r_dims.unOutputRows = std::stoul(dims_string[4]);
    r_dims.unTransformParametersColumns = std::stoul(dims_string[5]);
    r_dims.unTransformParametersRows = std::stoul(dims_string[6]);

    return r_dims;

}

eAct_func extract_act_function(std::vector<std::string>& act_string)
{
    assert(act_string.size() == 1);

    eAct_func r_act_func;

    assert(stringToActFunc.find(act_string[0]) != stringToActFunc.end());
    r_act_func = stringToActFunc[act_string[0]];

    return r_act_func;
}

float extract_act_param(std::vector<std::string>& act_string)
{
    assert(act_string.size() == 1);

    float r_act_param;

    r_act_param = std::stof(act_string[0]);

    return r_act_param;
}

void BaseLayer::extract_apply_biases(std::vector<std::string>& biases_string)
{
    assert(biases_string.size() == m_unNumNodes);

    for(uint i = 0; i < m_unNumNodes; i++)
    {
        set_node_bias(std::stof(biases_string[i]), i);
    }
}

void BaseLayer::extract_apply_transform_param(std::vector<std::string>& wt_string)
{
    assert(wt_string.size() == (m_Dimensions.unTransformParametersRows * m_Dimensions.unTransformParametersColumns * m_Dimensions.unNoTransformParameterMtx));

    m_unTransformMatrixSize = wt_string.size();
    m_pfTransformParameters = new float[m_unTransformMatrixSize];

    for(uint i = 0; i < m_unTransformMatrixSize; i++)
    {
        m_pfTransformParameters[i] = std::stof(wt_string[i]);
    }
}

void BaseLayer::apply_parsed_load_data(eLayer_type t_layer_type, std::vector<std::pair<std::string, std::vector<std::string>>>& parsed_lines)
{

    sLayer_Dimensions dims;
    eAct_func act_func;
    float act_param = 0.0f;

    bool dims_set = false;
    bool act_func_set = false;
    bool act_param_set = false;

    for(uint i = 0; i < parsed_lines.size(); i++)
    {
        if(parsed_lines[i].first == "dims:")
        {
            dims = extract_dims(parsed_lines[i].second);
            dims_set = true;
        }
        else if(parsed_lines[i].first == "act_func:")
        {
            act_func = extract_act_function(parsed_lines[i].second);
            act_func_set = true;
        }
        else if(parsed_lines[i].first == "act_param:")
        {
            act_param = extract_act_param(parsed_lines[i].second);
            act_param_set = true;
        }
        else
        {
            //ignore
        }
    }

    assert(dims_set);
    assert(act_func_set);
    assert(act_param_set);

    setup_layer(t_layer_type, dims, act_func, act_param);

    for(uint i = 0; i < parsed_lines.size(); i++)
    {
        if(parsed_lines[i].first == "biases:")
        {
            extract_apply_biases(parsed_lines[i].second);
        }
        else if(parsed_lines[i].first == "transform_mtx:")
        {
            if(parsed_lines[i].second.size())
            {
                extract_apply_transform_param(parsed_lines[i].second);
            }            
        }
        else
        {
            //ignore
        }
    }

}

void BaseLayer::setup_layer(eLayer_type t_layer_type, sLayer_Dimensions t_dims, eAct_func eActFunc, float actParam1)
{
    m_layer_type = t_layer_type;

    validate_input_data_layers(t_layer_type, t_dims);
    
    m_Dimensions = t_dims;

    m_unNumNodes = m_Dimensions.unOutputRows * m_Dimensions.unOutputColumns;

    m_eActFunc = eActFunc;
    m_actParam1 = actParam1;

    m_pfValues = new float[m_unNumNodes];
    m_pfBiases = new float[m_unNumNodes];
    m_pfDeltas = new float[m_unNumNodes];

    setup_activation_function(m_eActFunc);

    setup_accelerator();

}

void BaseLayer::SetPreviousNextLayers(BaseLayer* prevLyr, BaseLayer* nxtLyr)
{
    m_pPrevLyr = prevLyr;
    m_pNextLyr = nxtLyr;

    if((m_pfTransformParameters == nullptr) && (m_layer_type != eLayer_type::INPUT))
    {
        m_unTransformMatrixSize = m_Dimensions.unTransformParametersRows * m_Dimensions.unTransformParametersColumns * m_Dimensions.unNoTransformParameterMtx;
        m_pfTransformParameters = new float[m_unTransformMatrixSize];
    }

    m_bPrevNxtLyrsSet = true;
}

std::string BaseLayer::get_serialized_biases_data()
{
    std::ostringstream ss;
    ss << "biases: ";
    for(uint i = 0; i < m_unNumNodes; i++)
    {
        ss << m_pfBiases[i] << " ";
    }
    ss << "\n";

    return ss.str();
}

std::string BaseLayer::get_serialized_transform_mtx_data()
{
    std::ostringstream ss;
    ss << "transform_mtx: ";
    for(uint i = 0; i < m_unTransformMatrixSize; i++)
    {
        ss << m_pfTransformParameters[i] << " ";
    }
    ss << "\n";
    
    return ss.str();
}

sLayer_Parsed_Dim BaseLayer::get_prev_layer_parsed_output_dims()
{
    return get_parsed_dims(GetPreviousLayer()->get_layer_dimensions());
}

void BaseLayer::setup_accelerator()
{
#ifdef OPENCL_ACC
    m_pAccelerator = new OpenclAccelerator(this);
#else
    m_pAccelerator = new CpuAccelerator(this);
#endif
}

void BaseLayer::setup_activation_function(eAct_func eActFunc)
{
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
            assert(0); //unknown activation function
            break;
    }
}