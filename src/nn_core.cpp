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

#include <string.h>
#include <stdio.h>

#include <cassert>

static const char* static_NNDumpFilePath = "./"; //dump file path

static int static_nDumpFileNum = 0; //number postfix fro dump files



struct Batch_Training_Instance_data{
    float* deltas;
    bool bCorrectPredict;
};

struct Batch_Training_Args{
    float *in;
    float *out;
    Batch_Training_Instance_data *sBData;
};

static Batch_Training_Args ** args = nullptr;

// std::mutex argsMutex;



NeuralNet::NeuralNet(nnInitData* initData):
m_ppLys(nullptr)
{

    srand(time(NULL));

    Set_Init_Data(initData);
    
}

NeuralNet::NeuralNet(NeuralNet* other)
{
    nnInitData *ret = new nnInitData(m_unNumLys);

    //debug
    static uint i = 1;

    other->Get_Init_Data(ret);

    ret->ID = i;
    i++;

    Set_Init_Data(ret);
    populateWeightsAndBiasesWithExistingNN(other);

    delete ret;
}

NeuralNet::NeuralNet(const char* fileName)
{
    FILE* file = fopen(fileName, "r"); 
	if (file == nullptr) 
	{ 
		perror("Failed to open file for reading"); 
		return; 
	} 
	
    uint i = 0;
    uint j = 0;

    uint temp_numLys = 0;
    int temp_int = 0;
    float temp_float = 0.0f;

    //load init data
    fscanf(file, "%d", &temp_numLys);

    nnInitData* temp_initData = new nnInitData(temp_numLys);

    temp_initData->unNoLys = temp_numLys;

    for(i = 0; i < temp_numLys; i++)
    {
        fscanf(file, "%d", &(temp_initData->layer_dimensions[i].unInputRows));
        fscanf(file, "%d", &(temp_initData->layer_dimensions[i].unInputColumns));
        fscanf(file, "%d", &(temp_initData->layer_dimensions[i].unOutputRows));
        fscanf(file, "%d", &(temp_initData->layer_dimensions[i].unOutputColumns));
        fscanf(file, "%d", &(temp_initData->layer_dimensions[i].unTransformParametersRows));
        fscanf(file, "%d", &(temp_initData->layer_dimensions[i].unTransformParametersColumns));
        fscanf(file, "%d", &(temp_initData->layer_dimensions[i].unNoTransformParameterMtx));
    }

    for(i = 0; i < temp_numLys; i++)
    {
        // fscanf(file, "%d", &(temp_initData->eAct_Funcs[i]));
        fscanf(file, "%d", &temp_int);
        temp_initData->eAct_Funcs[i] = (eAct_func)temp_int;
    }

    // fscanf(file, "%d", &temp_int);
    // temp_initData->eAct_Func = (eAct_func)temp_int;
    fscanf(file, "%f", &temp_initData->fLearningRate);
    fscanf(file, "%d", &temp_int);
    temp_initData->eOpt = (eOptimizers)temp_int;
    fscanf(file, "%d", &temp_int);
    temp_initData->eLossFunc = (eLossFuncs)temp_int;

    fscanf(file, "%f", &temp_float);
    temp_initData->optParam1 = temp_float;
    fscanf(file, "%f", &temp_float);
    temp_initData->optParam2 = temp_float;
    fscanf(file, "%f", &temp_float);
    temp_initData->optParam3 = temp_float;

    for(i = 0; i < temp_numLys; i++)
    {
        fscanf(file, "%f", &(temp_initData->actParam1[i]));
    }

    // fscanf(file, "%f", &temp_float);
    // temp_initData->actParam1 = temp_float;

    fscanf(file, "%f", &temp_float);
    temp_initData->lossParam1 = temp_float;

    //set nn with temp init data
    Set_Init_Data(temp_initData);

    delete temp_initData;

    //store bias values
	for(i = 0; i < temp_numLys; ++i) 
	{ 
        for(j = 0; j < m_ppLys[i]->get_num_nodes(); j++)
        {
            fscanf(file, "%f", &temp_float);
            m_ppLys[i]->set_node_bias(temp_float, j);
        }
		 
	} 

    //store weight values
    for(i = 1; i < m_unNumLys; ++i) 
	{ 
        for(j = 0; j < m_ppLys[i]->get_transform_matrix_parameter_size(); j++)
        {
            fscanf(file, "%f", &temp_float);
            m_ppLys[i]->set_transform_matrix_parameter(j, temp_float);
        }
		 
	} 


	fclose(file);
}

void NeuralNet::Get_Init_Data(nnInitData *ret)
{
    ret->unNoLys = m_unNumLys;
    for(uint i = 0; i < m_unNumLys; i++)
    {
        ret->layer_dimensions[i] = m_ppLys[i]->get_layer_dimensions();


        ret->eAct_Funcs[i] = m_ppLys[i]->get_act_func();
        ret->actParam1[i] = m_ppLys[i]->get_act_param();
    }
    
    ret->fLearningRate = m_fLearningRate;
    ret->eOpt = m_eOpt;
    ret->eLossFunc = m_eLossFunc;

    ret->optParam1 = m_optParam1;
    ret->optParam2 = m_optParam2;
    ret->optParam3 = m_optParam3;
    
    ret->lossParam1 = m_lossParam1;
    // ret->ID = nn_id;

}

void NeuralNet::Set_Init_Data(nnInitData* other_initData)
{
    m_unNumLys = other_initData->unNoLys;
    
    m_fLearningRate = other_initData->fLearningRate;
    // nn_id = other_initData->ID;
    m_eOpt = other_initData->eOpt;

    m_eLossFunc = other_initData->eLossFunc;

    if(m_ppLys)
    {
        delete [] m_ppLys;
    }

    SetupLayersAndWeightMatrices(other_initData->e_layer_type, other_initData->layer_dimensions, other_initData->eAct_Funcs, other_initData->actParam1, other_initData->lossParam1);

    switch(m_eOpt)
    {
        case eOptimizers::SGD:
            m_pOptimizer = new StochasticGradientDescent(this);
            break;
        case eOptimizers::RMSPROP:
            m_pOptimizer = new RMSProp(this, other_initData->optParam1 != 0.0f ? other_initData->optParam1 : RMS_PROP_DEFAULT_BETA, other_initData->optParam2 != 0.0f ? other_initData->optParam2 : RMS_PROP_DEFAULT_EPSILON);
            break;
        case eOptimizers::ADAM:
            m_pOptimizer = new ADAMOPT(this, other_initData->optParam1 != 0.0f ? other_initData->optParam1 : ADAM_DEFAULT_BETA1, other_initData->optParam2 != 0.0f ? other_initData->optParam2 : ADAM_DEFAULT_BETA1, other_initData->optParam3 != 0.0f ? other_initData->optParam3 : ADAM_DEFAULT_EPSILON);
            break;
        default:
            m_pOptimizer = new StochasticGradientDescent(this);
            break;
    }

    //store param values locally
    m_optParam1 = other_initData->optParam1;
    m_optParam2 = other_initData->optParam2;
    m_optParam3 = other_initData->optParam3;

    m_lossParam1 = other_initData->lossParam1;

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

void NeuralNet::SetupLayersAndWeightMatrices(std::vector<eLayer_type>& layer_types, std::vector<sLayer_Dimensions>& dims, std::vector<eAct_func>& actFuncs, std::vector<float>& actParam1, float lossParam)
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
            
            case eLayer_type::DENSE:
            default:
                m_ppLys[i] = new DenseLayer(dims[i], actFuncs[i], actParam1[i]);
                break;
        }

        if(i != 0)
        {
            m_unTotalCorrectableNodes += dims[i].unOutputRows * dims[i].unOutputColumns;
        }

    }

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
                    m_pLossFunc = new HuberLoss(m_ppLys[i], lossParam != 0.0f ? lossParam : HUBER_DEFAULT_DELTA);
                    break;
                case eLossFuncs::CCE:
                    m_pLossFunc = new CompetitiveCrossEntropyLoss(m_ppLys[i]);
                    break;
                case eLossFuncs::BCE:
                    m_pLossFunc = new BinaryCrossEntropyLoss(m_ppLys[i]);
                    break;
                default:
                    m_pLossFunc = new MeanSquaredError(m_ppLys[i]);
                    break;
            }
            
        }
        
    }
    
}


bool NeuralNet::do_forward_pass(float* pfInputArr)
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

float NeuralNet::calculate_error(float* pfExpOut, float* pfError)
{
    float fRet = 0;

    fRet = m_pLossFunc->apply_loss_func(pfExpOut);

    return fRet;

}

bool NeuralNet::do_backward_pass(float* pfExpOut)
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


bool NeuralNet::Train(float* in, float* out)
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
        
        args[i]->in = new float[inputLayerSz];
        
        args[i]->out = new float[outputLayerSz];
        
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
        delete [] args[i]->out;
        delete [] args[i]->in;
        delete args[i];
    }

    delete [] args;
}

void NeuralNet::Populate_Batch_Processing_Args(float** in, float** out, uint numIn)
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


uint NeuralNet::Train_batch(float** in, float** out, uint numIn)
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



bool NeuralNet::isCorrectPrediction(float* pfOut)
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

bool NeuralNet::Test(float* pfIn, float* pfOut)
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

void NeuralNet::SaveNN(const char* fileName)
{
    FILE* file = fopen(fileName, "w"); 
	if (file == nullptr) 
	{ 
		perror("Failed to open file for writing"); 
		return; 
	}

    uint i = 0;
    uint j = 0;

    //store init data
    fprintf(file, "%d ", m_unNumLys);
    for(i = 0; i < m_unNumLys; i++)
    {
        fprintf(file, "%d ", m_ppLys[i]->get_num_nodes());
    }

    for(i = 0; i < m_unNumLys; i++)
    {
        fprintf(file, "%d ", m_ppLys[i]->get_act_func());
    }

    // fprintf(file, "%d ", (int)m_eActFunc);
    fprintf(file, "%f ", m_fLearningRate);
    fprintf(file, "%d ", (int)m_eOpt);
    fprintf(file, "%d ", (int)m_eLossFunc);

    fprintf(file, "%f ", m_optParam1);
    fprintf(file, "%f ", m_optParam2);
    fprintf(file, "%f ", m_optParam3);

    for(i = 0; i < m_unNumLys; i++)
    {
        fprintf(file, "%f ", m_ppLys[i]->get_act_param());
    }

    // fprintf(file, "%f ", m_actParam1);

    fprintf(file, "%f ", m_lossParam1);


    //store bias values
	for(i = 0; i < m_unNumLys; ++i) 
	{ 
        for(j = 0; j < m_ppLys[i]->get_num_nodes(); j++)
        {
            fprintf(file, "%f ", m_ppLys[i]->get_node_bias_idx(j));
        }
		 
	} 

    //store weight values
    for(i = 1; i < m_unNumLys; ++i) 
	{ 
        for(j = 0; j < m_ppLys[i]->get_transform_matrix_parameter_size(); j++)
        {
            fprintf(file, "%f ", m_ppLys[i]->get_transform_matrix_parameter(j));
        }
		 
	} 
	fclose(file);
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

