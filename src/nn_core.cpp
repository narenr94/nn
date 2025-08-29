#include "nn_core.h"

//activation functions
#include "sigmoidActFunc.h"
#include "reluActFunc.h"
#include "leakyReluActFunc.h"
#include "tanhActFunc.h"

//Optimizers
#include "stochasticGradientDescent.h"
#include "rmsprop.h"
#include "adam.h"

//Loss functions
#include "meanSquaredError.h"
#include "meanAbsoluteError.h"
#include "huberLoss.h"
#include "binaryCrossEntropyLoss.h"
#include "competitiveCrossEntropyLoss.h"

//Layers
#include "denseLayer.h"
#include "convLayer.h"
#include "inputLayer.h"
#include "poolingLayer.h"

#include <string.h>
#include <stdio.h>
#include <cassert>

std::map<eLayer_type, std::string> layerTypeToString = {
        {INPUT, "INPUT"},
        {CONV, "CONV"},
        {POOLING, "POOLING"},
        {DENSE, "DENSE"}
    };

std::map<std::string, eLayer_type> stringToLayerType = {
        {"INPUT", INPUT},
        {"CONV", CONV},
        {"POOLING", POOLING},
        {"DENSE", DENSE}
    };

std::map<eLossFuncs, std::string> lossFuncToString = {
        {MSE, "MSE"},
        {MAE, "MAE"},
        {HUBER, "HUBER"},
        {BCE, "BCE"},
        {CCE, "CCE"}
    };

std::map<std::string, eLossFuncs> stringToLossFunc = {
        {"MSE", MSE},
        {"MAE", MAE},
        {"HUBER", HUBER},
        {"BCE", BCE},
        {"CCE", CCE}
    };

std::map<eOptimizers, std::string> optimizerToString = {
        {SGD, "SGD"},
        {RMSPROP, "RMSPROP"},
        {ADAM, "ADAM"}
    };

std::map<std::string, eOptimizers> stringToOptimizer = {
        {"SGD", SGD},
        {"RMSPROP", RMSPROP},
        {"ADAM", ADAM}
    };


static const char* static_NNDumpFilePath = "./"; //dump file path

static int static_nDumpFileNum = 0; //number postfix fro dump files

struct Batch_Training_Instance_data{
    float* deltas;
    bool bCorrectPredict;
};

struct Batch_Training_Args{
    std::vector<float> in;
    std::vector<float> out;
    Batch_Training_Instance_data *sBData;
};

static Batch_Training_Args ** args = nullptr;

// std::mutex argsMutex;

NeuralNet::NeuralNet(nnInitData& initData):
m_ppLys(nullptr)
{

    srand(time(NULL));

    Set_Init_Data(initData);
    
}

NeuralNet::NeuralNet(NeuralNet* other)
{
    nnInitData ret = other->Get_Init_Data();

    //debug
    static uint i = 1;

    ret.ID = i;
    i++;

    Set_Init_Data(ret);
    populateWeightsAndBiasesWithExistingNN(other);

}

sNN_General_Data get_general_nn_data(std::vector<std::pair<std::string, std::vector<std::string>>> parsed_lines)
{
    sNN_General_Data ret_data;
    bool m_unNumLys_set = false;
    bool m_fLearningRate_set = false;
    bool m_eOpt_set = false;
    bool m_eLossFunc_set = false;
    bool layer_types_set = false;

    for(uint i = 0; i < parsed_lines.size(); i++)
    {
        if(parsed_lines[i].first == "m_unNumLys:")
        {
            assert(parsed_lines[i].second.size() == 1);
            ret_data.m_unNumLys = std::stoul(parsed_lines[i].second[0]);
            m_unNumLys_set = true;
        }
        else if(parsed_lines[i].first == "m_fLearningRate:")
        {
            assert(parsed_lines[i].second.size() == 1);
            ret_data.m_fLearningRate = std::stof(parsed_lines[i].second[0]);
            m_fLearningRate_set = true;
        }
        else if(parsed_lines[i].first == "m_eOpt:")
        {
            assert(parsed_lines[i].second.size() == 1);
            assert(stringToOptimizer.find(parsed_lines[i].second[0]) != stringToOptimizer.end());
            ret_data.m_eOpt = stringToOptimizer[parsed_lines[i].second[0]];
            m_eOpt_set = true;
        }
        else if(parsed_lines[i].first == "m_eLossFunc:")
        {
            assert(parsed_lines[i].second.size() == 1);
            assert(stringToLossFunc.find(parsed_lines[i].second[0]) != stringToLossFunc.end());
            ret_data.m_eLossFunc = stringToLossFunc[parsed_lines[i].second[0]];
            m_eLossFunc_set = true;
        }
        else if(parsed_lines[i].first == "m_optParam1:")
        {
            assert(parsed_lines[i].second.size() == 1);
            ret_data.m_optParam[0] = std::stof(parsed_lines[i].second[0]);
        }
        else if(parsed_lines[i].first == "m_optParam2:")
        {
            assert(parsed_lines[i].second.size() == 1);
            ret_data.m_optParam[1] = std::stof(parsed_lines[i].second[0]);
        }
        else if(parsed_lines[i].first == "m_optParam3:")
        {
            assert(parsed_lines[i].second.size() == 1);
            ret_data.m_optParam[2] = std::stof(parsed_lines[i].second[0]);
        }
        else if(parsed_lines[i].first == "m_lossParam1:")
        {
            assert(parsed_lines[i].second.size() == 1);
            ret_data.m_lossParam = std::stof(parsed_lines[i].second[0]);
        }
        else if(parsed_lines[i].first == "layer_types:")
        {
            assert(parsed_lines[i].second.size() == ret_data.m_unNumLys);
            for(uint j = 0; j < ret_data.m_unNumLys; j++)
            {
                assert(stringToLayerType.find(parsed_lines[i].second[j]) != stringToLayerType.end());
                ret_data.layer_types.push_back(stringToLayerType[parsed_lines[i].second[j]]);
            }
            layer_types_set = true;
        }
        else
        {
            //ignore
        }
    }

    assert(m_unNumLys_set);
    assert(m_fLearningRate_set);
    assert(m_eOpt_set);
    assert(m_eLossFunc_set);
    assert(layer_types_set);

    return ret_data;
}

NeuralNet::NeuralNet(std::string& fileName)
{

    std::string file_data = read_file(fileName);

    std::vector<std::string> blocks = split_by_delimiter(file_data);

    Set_Load_Data(blocks);

}

void NeuralNet::Set_Load_Data(std::vector<std::string> blocks)
{
    std::vector<std::string> lines = split_by_lines(blocks[0]);

    std::vector<std::pair<std::string, std::vector<std::string>>> parsed_lines;

    for(uint i = 0; i < lines.size(); i++)
    {
        parsed_lines.push_back(parse_line(lines[i]));
    }

    sNN_General_Data gen_data = get_general_nn_data(parsed_lines);

    Set_General_Data(gen_data);
    Set_Layer_Data(gen_data, blocks);
}

void NeuralNet::Set_Layer_Data(sNN_General_Data gen_data, std::vector<std::string> blocks)
{
    m_ppLys = new BaseLayer*[m_unNumLys];
    
    //create layers
    for(uint i = 0; i < m_unNumLys; i++)
    {
        switch(gen_data.layer_types[i])
        {
            case eLayer_type::INPUT:
                m_ppLys[i] = new InputLayer(blocks[i + 1]);
                break;

            case eLayer_type::CONV:
                m_ppLys[i] = new ConvLayer(blocks[i + 1]);
                break;

            case eLayer_type::POOLING:
                m_ppLys[i] = new PoolingLayer(blocks[i + 1]);
                break;
            
            case eLayer_type::DENSE:
                m_ppLys[i] = new DenseLayer(blocks[i + 1]);
                break;

            default:
                assert(0); //unknown layer type
                break;
        }

        if(i != 0)
        {
            m_unTotalCorrectableNodes += m_ppLys[i]->get_num_nodes();
        }

    }

    Set_Layer_Order(gen_data.m_lossParam);
    
}

void NeuralNet::Set_Layer_Order(float m_lossParam)
{
    //setup layers order
    for(uint i = 0; i < m_unNumLys; i++)
    {
        if((i != 0) && (i != (m_unNumLys - 1))) //hidden layers
        {
            m_ppLys[i]->SetPreviousNextLayers(m_ppLys[i - 1], m_ppLys[i + 1]);
        }
        else if(i == 0) //input layer
        {
            m_ppLys[i]->SetPreviousNextLayers(nullptr, m_ppLys[i + 1]);
        }
        else// output layer
        {
            m_ppLys[i]->SetPreviousNextLayers(m_ppLys[i - 1], nullptr);
            switch(m_eLossFunc)
            {
                case eLossFuncs::MSE:
                    m_pLossFunc = new MeanSquaredError(m_ppLys[i]);
                    break;
                case eLossFuncs::MAE:
                    m_pLossFunc = new MeanAbsoluteError(m_ppLys[i]);
                    break;
                case eLossFuncs::HUBER:
                    m_pLossFunc = new HuberLoss(m_ppLys[i], m_lossParam != 0.0f ? m_lossParam : HUBER_DEFAULT_DELTA);
                    break;
                case eLossFuncs::CCE:
                    m_pLossFunc = new CompetitiveCrossEntropyLoss(m_ppLys[i]);
                    break;
                case eLossFuncs::BCE:
                    m_pLossFunc = new BinaryCrossEntropyLoss(m_ppLys[i]);
                    break;
                default:
                    assert(0); //unknown loss function
                    break;
            }
            
        }
        
    }
}

void NeuralNet::Set_General_Data(sNN_General_Data gen_data)
{
    m_unNumLys = gen_data.m_unNumLys;
    
    m_fLearningRate = gen_data.m_fLearningRate;
    // nn_id = other_initData->ID;
    m_eOpt = gen_data.m_eOpt;

    m_eLossFunc = gen_data.m_eLossFunc;

    if(m_ppLys)
    {
        delete [] m_ppLys;
    }

    //SetupLayersAndWeightMatrices
    
    switch(m_eOpt)
    {
        case eOptimizers::SGD:
            m_pOptimizer = new StochasticGradientDescent(this);
            break;
        case eOptimizers::RMSPROP:
            m_pOptimizer = new RMSProp(this, gen_data.m_optParam[0] != 0.0f ? gen_data.m_optParam[0] : RMS_PROP_DEFAULT_BETA, gen_data.m_optParam[1] != 0.0f ? gen_data.m_optParam[1] : RMS_PROP_DEFAULT_EPSILON);
            break;
        case eOptimizers::ADAM:
            m_pOptimizer = new ADAMOPT(this, gen_data.m_optParam[0] != 0.0f ? gen_data.m_optParam[0] : ADAM_DEFAULT_BETA1, gen_data.m_optParam[1] != 0.0f ? gen_data.m_optParam[1] : ADAM_DEFAULT_BETA1, gen_data.m_optParam[2] != 0.0f ? gen_data.m_optParam[2] : ADAM_DEFAULT_EPSILON);
            break;
        default:
            assert(0); //uknown optimizer
            break;
    }

    //store param values locally
    m_optParam1 = gen_data.m_optParam[0];
    m_optParam2 = gen_data.m_optParam[1];
    m_optParam3 = gen_data.m_optParam[2];

    m_lossParam1 = gen_data.m_lossParam;
}

nnInitData NeuralNet::Get_Init_Data()
{
    nnInitData ret;
    ret.unNoLys = m_unNumLys;
    for(uint i = 0; i < m_unNumLys; i++)
    {
        ret.layer_dimensions.emplace_back(m_ppLys[i]->get_layer_dimensions());
        ret.eAct_Funcs.emplace_back(m_ppLys[i]->get_act_func());
        ret.actParam1.emplace_back(m_ppLys[i]->get_act_param());
        ret.e_layer_type.emplace_back(m_ppLys[i]->get_layer_type());
        if(m_ppLys[i]->get_layer_type() != eLayer_type::POOLING)
        {
            ret.ePoolingType.emplace_back(ePooling_type::NA);
        }
        else
        {
            ret.ePoolingType.emplace_back(static_cast<ePooling_type>(static_cast<int>(m_ppLys[i]->get_act_param())));
        }
    }
    
    ret.fLearningRate = m_fLearningRate;
    ret.eOpt = m_eOpt;
    ret.eLossFunc = m_eLossFunc;

    ret.optParam[0] = m_optParam1;
    ret.optParam[1] = m_optParam2;
    ret.optParam[2] = m_optParam3;
    
    ret.lossParam = m_lossParam1;
    // ret->ID = nn_id;
    return ret;

}

void NeuralNet::Set_Init_Data(nnInitData& other_initData)
{
    m_unNumLys = other_initData.unNoLys;
    
    m_fLearningRate = other_initData.fLearningRate;
    // nn_id = other_initData->ID;
    m_eOpt = other_initData.eOpt;

    m_eLossFunc = other_initData.eLossFunc;

    if(m_ppLys)
    {
        delete [] m_ppLys;
    }

    SetupLayersAndWeightMatrices(other_initData.e_layer_type, other_initData.layer_dimensions, other_initData.eAct_Funcs, other_initData.actParam1, other_initData.lossParam, other_initData.ePoolingType);

    switch(m_eOpt)
    {
        case eOptimizers::SGD:
            m_pOptimizer = new StochasticGradientDescent(this);
            break;
        case eOptimizers::RMSPROP:
            m_pOptimizer = new RMSProp(this, other_initData.optParam[0] != 0.0f ? other_initData.optParam[0] : RMS_PROP_DEFAULT_BETA, other_initData.optParam[1] != 0.0f ? other_initData.optParam[1] : RMS_PROP_DEFAULT_EPSILON);
            break;
        case eOptimizers::ADAM:
            m_pOptimizer = new ADAMOPT(this, other_initData.optParam[0] != 0.0f ? other_initData.optParam[0] : ADAM_DEFAULT_BETA1, other_initData.optParam[1] != 0.0f ? other_initData.optParam[1] : ADAM_DEFAULT_BETA1, other_initData.optParam[2] != 0.0f ? other_initData.optParam[2] : ADAM_DEFAULT_EPSILON);
            break;
        default:
            assert(0); //uknown optimizer
            break;
    }

    //store param values locally
    m_optParam1 = other_initData.optParam[0];
    m_optParam2 = other_initData.optParam[1];
    m_optParam3 = other_initData.optParam[2];

    m_lossParam1 = other_initData.lossParam;

}

NeuralNet::~NeuralNet()
{
    if(m_ppLys)
    {
        for(uint i = 0; i < m_unNumLys; i++)
        {
            delete m_ppLys[i];
        }
        delete [] m_ppLys;
    }

    if(initBatchTrain)
    {
        for(uint i  = 0; i < m_batchSz; i++)
        {
            
            delete m_batch_nns[i];            

        }
        delete [] m_batch_nns;
    }

    if(m_pOptimizer)
    {
        delete m_pOptimizer;
    }

    if(m_pLossFunc)
    {
        delete m_pLossFunc;
    }
}

void NeuralNet::SetupLayersAndWeightMatrices(std::vector<eLayer_type>& layer_types, std::vector<sLayer_Dimensions>& dims, std::vector<eAct_func>& actFuncs, std::vector<float>& actParam1, float lossParam, std::vector<ePooling_type>& ePoolingType)
{
    m_ppLys = new BaseLayer*[m_unNumLys];
    
    //create layers
    for(uint i = 0; i < m_unNumLys; i++)
    {
        switch(layer_types[i])
        {
            case eLayer_type::INPUT:
                m_ppLys[i] = new InputLayer(dims[i], actFuncs[i], actParam1[i]);
                break;

            case eLayer_type::CONV:
                m_ppLys[i] = new ConvLayer(dims[i], actFuncs[i], actParam1[i]);
                break;
            
            case eLayer_type::POOLING:
                m_ppLys[i] = new PoolingLayer(dims[i], ePoolingType[i]);
                break;
            
            case eLayer_type::DENSE:
                m_ppLys[i] = new DenseLayer(dims[i], actFuncs[i], actParam1[i]);
                break;
            
            default:
                assert(0); //unknown layer type
                break;
        }

        if(i != 0)
        {
            m_unTotalCorrectableNodes += dims[i].unOutputRows * dims[i].unOutputColumns;
        }

    }

    Set_Layer_Order(lossParam);
    
}


bool NeuralNet::do_forward_pass(std::vector<float>& pfInputArr)
{

    bool bRet = false;

    //set input layer nodes
    m_ppLys[INPUT_LAYER_ID]->set_all_node_values(pfInputArr);

    uint i = 0;

    for(i = (INPUT_LAYER_ID + 1); i < m_unNumLys; i++)
    {
        m_ppLys[i]->do_forwardpass_to_current_layer();
    }
    
    bRet = true;

    return bRet;

}




void NeuralNet::populate_weights(uint unIdx, float* pfValues)
{
    assert(unIdx != 0);

    m_ppLys[unIdx]->set_all_transform_matrix_parameter(pfValues);

}

bool NeuralNet::populate_nodes_bias(uint unLyrIdx, float* pfBias)
{
    assert((unLyrIdx < m_unNumLys) && (unLyrIdx > 0));

    bool bRet = false;

    bRet = m_ppLys[unLyrIdx]->set_all_node_biases(pfBias);

    return bRet;
}

float NeuralNet::calculate_error(std::vector<float>& pfExpOut, std::vector<float>& pfError)
{
    float fRet = 0;

    fRet = m_pLossFunc->apply_loss_func(pfExpOut);

    return fRet;

}

bool NeuralNet::do_backward_pass(std::vector<float>& pfExpOut)
{
    bool bRet = false;

    //calculate error for every layer's node. except input layer

    /*
        for output layer nodes:
            error = der_act_func(actual value) * (exp_value - actual_value)
        for hidden layer nodes:
            error = der_act_func(actual value) * (sum(weights_leading_out_of_node * error_of_node_it_is_reaching))
    */
    for(int i = (m_unNumLys - 1); i >= INPUT_LAYER_ID; i--)
    {
        if(i != (m_unNumLys - 1))
        {
            m_ppLys[i]->do_backwardpass_to_previous_layer();
        }
        else
        {
            m_ppLys[i]->do_backwardpass_to_previous_layer_output_layer(pfExpOut, m_pLossFunc);
        }
    }
    

    m_pOptimizer->correct_transform_parameters_and_biases();

    bRet = true;

    return bRet;

}


bool NeuralNet::Train(std::vector<float>& in, std::vector<float>& out)
{
    bool bRet = false;

    do_forward_pass(in);

    bRet = isCorrectPrediction(out);

    do_backward_pass(out);

    return bRet;
}

void NeuralNet::MergeBiasAndWeights(uint i)
{
    //merge bias
    for(uint j = 1; j < m_unNumLys; j++)
    {
        for(uint k = 0; k < m_ppLys[j]->get_num_nodes(); k++)
        {
            m_ppLys[j]->set_node_bias(m_ppLys[j]->get_node_bias_idx(k) + m_batch_nns[i]->GetBias(j, k), k);
        }
    }
    //merge weights
    for(uint j = 1; j < m_unNumLys ; j++)
    {
        for(uint k = 0; k < m_ppLys[j]->get_transform_matrix_parameter_size(); k++)
        {
            m_ppLys[j]->set_transform_matrix_parameter(k, m_ppLys[j]->get_transform_matrix_parameter(k) + m_batch_nns[i]->GetWeight(j, k));
        }
    }
}

void NeuralNet::Batch_Training(uint i) 
{ 
    do_forward_pass(args[i]->in);

    args[i]->sBData->bCorrectPredict = isCorrectPrediction(args[i]->out);    

    do_backward_pass(args[i]->out);
    
    
}


void Setup_Batch_Processing_Args(uint numIn, uint TotalCorrectableNodes, uint inputLayerSz, uint outputLayerSz)
{
    
    args = new Batch_Training_Args* [numIn]; 

    
    for(uint i = 0; i < numIn; i++)
    {
        args[i] = new Batch_Training_Args();
        
        args[i]->in = std::vector<float>(inputLayerSz);
        
        args[i]->out = std::vector<float>(outputLayerSz);
        
        args[i]->sBData = new Batch_Training_Instance_data();        
        
        args[i]->sBData->deltas = new float[TotalCorrectableNodes];
        
    }

    
}

void Release_Args(uint numIn)
{
    for(uint i = 0; i < numIn; i++)
    {
        delete [] args[i]->sBData->deltas;
        delete args[i]->sBData;
        args[i]->out.clear();
        args[i]->in.clear();
        delete args[i];
    }

    delete [] args;
}

void NeuralNet::Populate_Batch_Processing_Args(std::vector<std::vector<float>>& in, std::vector<std::vector<float>>& out, uint numIn)
{
    for(uint i =0; i < numIn; i++)
    {
        for(uint j = 0; j < m_ppLys[0]->get_num_nodes(); j++)
        {
            args[i]->in[j] = in[i][j];
        }
        for(uint k = 0; k < m_ppLys[m_unNumLys - 1]->get_num_nodes(); k++)
        {
            args[i]->out[k] = out[i][k];
        }

    }
}

void NeuralNet::Init_Batch_Training(uint batchSz)
{
    if(initBatchTrain)
    {

        delete [] m_batch_nns;

    }

    m_batch_nns = new NeuralNet*[batchSz];

    for(uint i  = 0; i < batchSz; i++)
    {
        
        m_batch_nns[i] = new NeuralNet(this);
        

    }

    m_batchSz = batchSz;

    initBatchTrain = true;

}


uint NeuralNet::Train_batch(std::vector<std::vector<float>>& in, std::vector<std::vector<float>>& out, uint numIn)
{
    // printf("\nEntered Train_batch\n");
    // fflush(stdout);
    std::thread **threads = new std::thread* [numIn];
    uint bRet = 0;

    
    Setup_Batch_Processing_Args(numIn, m_unTotalCorrectableNodes, m_ppLys[0]->get_num_nodes(),m_ppLys[m_unNumLys  -1]->get_num_nodes());
    
    Populate_Batch_Processing_Args(in, out, numIn);
    
    // Create threads dynamically
    for(uint i = 0; i < numIn; i++)
    {
        //thread th4(&Base::foo, &b);
        threads[i] = new std::thread(&NeuralNet::Batch_Training, m_batch_nns[i], i);              
        
    }
    

    for (uint i = 0; i < numIn; i++) 
    { 
        if (threads[i]->joinable())
        { 
            threads[i]->join(); 

            MergeBiasAndWeights(i);
            if(args[i]->sBData->bCorrectPredict)
            {
                bRet++;
            }
            
        } 
        delete threads[i];
        
        
    }

    // apply_delats_to_weights_and_biases_batch_training(deltas);
       
    m_pOptimizer->correct_transform_parameters_and_biases();
    
    delete [] threads; 
    Release_Args(numIn);
    args = nullptr;

    return bRet;
}

void NeuralNet::populateWeightsAndBiasesWithRandomNumbers()
{
    uint i = 0;

    for(i = 0; i < m_unNumLys; i++)
    {
        if(i != 0)
        {
            m_ppLys[i]->populate_transform_matrix_parameter_with_random_numbers();
        }
        m_ppLys[i]->populateBiasesWithRandomNumbers();
    }
}

float NeuralNet::GetBias(uint LayerID, uint NodeID)
{
    return m_ppLys[LayerID]->get_node_bias_idx(NodeID);
}

uint NeuralNet::GetSzLayer(uint LayerID)
{
    return m_ppLys[LayerID]->get_num_nodes();
}

uint NeuralNet::GetSzMtx(uint MtxID)
{
    assert(MtxID > 0);
    return m_ppLys[MtxID]->get_transform_matrix_parameter_size();
}

float NeuralNet::GetWeight(uint MtxID, uint Idx)
{
    assert(MtxID > 0);
    return m_ppLys[MtxID]->get_transform_matrix_parameter(Idx);
}


void NeuralNet::populateWeightsAndBiasesWithExistingNN(NeuralNet* other)
{
    uint i = 0;
    uint j = 0;
    uint sz = 0;

    for(i = 1; i < m_unNumLys; i++)
    {
        sz = other->GetSzLayer(i);
        for(j = 0; j < sz; j++)
        {
            m_ppLys[i]->set_node_bias(other->GetBias(i, j), j);
        }
    }
    for(i = 1; i < m_unNumLys; i++)
    {
        sz = other->GetSzMtx(i);
        for(j = 0; j < sz; j++)
        {
            m_ppLys[i]->set_transform_matrix_parameter(j, other->GetWeight(i, j));
        }
    }
}



bool NeuralNet::isCorrectPrediction(std::vector<float>& pfOut)
{
    bool bRet = false;

    if(m_ppLys[m_unNumLys - 1]->get_num_nodes() == 1) //cant get max if only one output exists, compare with 0.5f instead
    {
        if(pfOut[0] >= 0.5)
        {
            if(m_ppLys[m_unNumLys - 1]->get_node_value_idx(0) >= 0.5)
            {
                bRet = true;
            }
            
        }
        else
        {
            if(m_ppLys[m_unNumLys - 1]->get_node_value_idx(0) < 0.5)
            {
                bRet = true;
            }
        }
    }
    else
    {
        uint out_correct_label = 0;

        uint highest_out_label = 0;

        float highest_value = 0.0;

        uint i = 0;

        for(i = 0; i < m_ppLys[m_unNumLys - 1]->get_num_nodes(); i++)
        {
            if(pfOut[i] == 1.0)
            {
                out_correct_label = i;
            }

            if(m_ppLys[m_unNumLys - 1]->get_node_value_idx(i) > highest_value)
            {
                highest_value = m_ppLys[m_unNumLys - 1]->get_node_value_idx(i);
                highest_out_label = i;
            }

        }

        if(highest_out_label == out_correct_label)
        {
            bRet = true;
        }
    }

    

    return bRet;

}

bool NeuralNet::Test(std::vector<float>& pfIn, std::vector<float>& pfOut)
{

    bool ret = false;

    do_forward_pass(pfIn);

    ret = isCorrectPrediction(pfOut);    

    return ret;

}

uint NeuralNet::GetNumLys()
{
    return m_unNumLys;
}

void NeuralNet::SetBias(uint LayerID, uint NodeID, float val)
{
    m_ppLys[LayerID]->set_node_bias(val, NodeID);
}

float NeuralNet::GetLearningRate()
{
    return m_fLearningRate;
}

float NeuralNet::GetDelta(uint LayerID, uint NodeID)
{
    return m_ppLys[LayerID]->get_node_delta_idx(NodeID);
}

float NeuralNet::GetNodeVal(uint LayerID, uint NodeID)
{
    return m_ppLys[LayerID]->get_node_value_idx(NodeID);
}

void NeuralNet::SetWeight(uint MtxId, uint inIdx, uint outIdx, float val)
{
    assert(MtxId > 0);
    m_ppLys[MtxId]->set_transform_matrix_parameter(inIdx, outIdx, val);
}

void NeuralNet::SetWeight(uint MtxId, uint Idx, float val)
{
    assert(MtxId > 0);
    m_ppLys[MtxId]->set_transform_matrix_parameter(Idx, val);
}

void NeuralNet::Save_NN(std::string fileName)
{
    std::ostringstream ss;

    ss << "m_unNumLys: " << m_unNumLys << " \n";
    ss << "m_fLearningRate: " << m_fLearningRate << " \n";
    assert(optimizerToString.find(m_eOpt) != optimizerToString.end());
    ss << "m_eOpt: " << optimizerToString[m_eOpt] << " \n";
    ss << "m_eLossFunc: " << lossFuncToString[m_eLossFunc] << " \n";
    ss << "m_optParam1: " << m_optParam1 << " \n";
    ss << "m_optParam2: " << m_optParam2 << " \n";
    ss << "m_optParam3: " << m_optParam3 << " \n";
    ss << "m_lossParam1: " << m_lossParam1 << " \n";
    ss << "layer_types: ";
    for(uint i = 0; i < m_unNumLys; i++)
    {
        assert(layerTypeToString.find(m_ppLys[i]->get_layer_type()) != layerTypeToString.end());
        ss << layerTypeToString[m_ppLys[i]->get_layer_type()] << " ";
    }
    ss << "\n";

    for(uint i = 0; i < m_unNumLys; i++)
    {
        ss << "***\n";
        ss << m_ppLys[i]->get_serialized_save_data();
    }

    save_to_file(ss.str(), fileName);

}

void NeuralNet::Get_OutputLayer_Data(float* fVal)
{
    for(uint i = 0; i < m_ppLys[m_unNumLys - 1]->get_num_nodes(); i++)
    {
        fVal[i] = m_ppLys[m_unNumLys - 1]->get_node_value_idx(i);
    }
}

BaseLayer* NeuralNet::GetLayer(uint idx)
{
    return m_ppLys[idx];
}

const float* NeuralNet::GetMatrix(uint idx)
{
    assert(idx > 0);
    return m_ppLys[idx]->get_transform_matrix();
}

BaseLossFunction* NeuralNet::GetLossFunc()
{
    return m_pLossFunc;
}

eLayer_type NeuralNet::get_layer_type(uint idx)
{
    assert(idx < m_unNumLys);
    return m_ppLys[idx]->get_layer_type();
}

