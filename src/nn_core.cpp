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

#include <vector>

static const char* static_NNDumpFilePath = "./"; //dump file path

static int static_nDumpFileNum = 0; //number postfix fro dump files

/*
    list of activation functions
*/
static const char * static_const_parrActFuncStr[SIGMOID+1] =
{
    "RELU",
    "LEAKY_RELU",
    "TANH",
    "SIGMOID"
};

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



NeuralNet::NeuralNet(nnInitData* initData)
{
    srand(time(NULL));

    Set_Init_Data(initData);

    SetupLayersAndWeightMatrices(initData->unSzLys);
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

void NeuralNet::Get_Init_Data(nnInitData *ret)
{
    ret->unNoLys = m_unNumLys;
    for(uint i = 0; i < m_unNumLys; i++)
    {
        ret->unSzLys[i] = m_ppLys[i]->get_num_nodes();
    }
    ret->eAct_Func = m_eActFunc;
    ret->fLearningRate = m_fLearningRate;
    ret->eOpt = m_eOpt;
    // ret->ID = nn_id;

}

void NeuralNet::Set_Init_Data(nnInitData* other_initData)
{

    m_unNumLys = other_initData->unNoLys;
    m_eActFunc = other_initData->eAct_Func;
    switch(m_eActFunc)
    {
        case eAct_func::SIGMOID:
            m_pActFunc = new SigmoidActFunc();
            break;
        case eAct_func::RELU:
            m_pActFunc = new ReluActFunc();
            break;
        case eAct_func::LEAKY_RELU:
            m_pActFunc = new LeakyReluActFunc();
            break;
        case eAct_func::TANH:
            m_pActFunc = new TanhActFunc();
            break;
        default:
            m_pActFunc = new SigmoidActFunc();
    }
    m_fLearningRate = other_initData->fLearningRate;
    // nn_id = other_initData->ID;
    m_eOpt = other_initData->eOpt;

    

    

    if(m_bInitialized)
    {
        delete [] m_ppLys;
        delete [] m_ppWtMtcs;
    }

    

    SetupLayersAndWeightMatrices(other_initData->unSzLys);

    switch(m_eOpt)
    {
        case eOptimizers::SGD:
            m_pOptimizer = new StochasticGradientDescent(this);
            break;
        case eOptimizers::RMSPROP:
            m_pOptimizer = new RMSProp(this);
            break;
        case eOptimizers::ADAM:
            m_pOptimizer = new ADAMOPT(this);
            break;
        default:
            m_pOptimizer = new StochasticGradientDescent(this);
            break;
    }

    m_bInitialized = true;


}

NeuralNet::~NeuralNet()
{
    if(m_ppWtMtcs)
    {
        for(uint i = 0; i < (m_unNumLys - 1); i++)
        {
            delete m_ppWtMtcs[i];
        }
        delete [] m_ppWtMtcs;
    }
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
}


void NeuralNet::SetupLayersAndWeightMatrices(uint *sz)
{
    m_ppLys = new nn_layer*[m_unNumLys];
    m_ppWtMtcs = new nn_l2l_weight_matrix*[m_unNumLys - 1];
    for(uint i = 0; i < m_unNumLys; i++)
    {
        m_ppLys[i] = new nn_layer(sz[i]);
        if(i == INPUT_LAYER_ID)
        {
            m_ppLys[i]->set_layer_type(INPUT_LYR);
        }
        else if(i == (m_unNumLys -1))
        {
            m_ppLys[i]->set_layer_type(OUTPUT_LYR);
            m_unTotalCorrectableNodes += sz[i];
        }
        else
        {
            m_ppLys[i]->set_layer_type(HIDDEN_LYR);
            m_unTotalCorrectableNodes += sz[i];
        }

    }

    for(uint i = 0; i < (m_unNumLys - 1); i++)
    {
         m_ppWtMtcs[i] = new nn_l2l_weight_matrix(m_ppLys[i], m_ppLys[i + 1]);
    }

}


bool NeuralNet::do_forward_pass(float* pfInputArr)
{

    bool bRet = false;

    //check if NN is initalised
    if(!m_bInitialized)
    {
        return bRet;
    }

    //set input layer nodes
    m_ppLys[INPUT_LAYER_ID]->set_all_node_values(pfInputArr);

    uint i = 0;

    // for(i = 0; i < 784; i++)
    // {
    //     NNLOG_MIL("[%d]%f", i, m_lys[INPUT_LAYER_ID]->get_node_value_idx(i));
    // }

    for(i = 0; i < (m_unNumLys - 1); i++)
    {
        if(!do_forwardpass_to_next_layer(INPUT_LAYER_ID + i))
        {
            return bRet;
        }
    }
    
    bRet = true;

    return bRet;

}

bool NeuralNet::do_forwardpass_to_next_layer(uint unInLayerIdx)
{
    bool bRet = false;

    if(!m_bInitialized)
    {
        return bRet;
    }

    nn_layer* in_lyr = m_ppLys[unInLayerIdx];
    nn_layer* out_lyr = m_ppLys[unInLayerIdx + 1];

    nn_l2l_weight_matrix* curr_mtx_ptr = m_ppWtMtcs[unInLayerIdx];

    uint in_lyr_sz = in_lyr->get_num_nodes();
    uint out_lyr_sz = out_lyr->get_num_nodes();

    uint i = 0;
    uint j = 0;

    float sigma = 0;

    for(j = 0; j < out_lyr_sz; j++)
    {
        for(i = 0; i < in_lyr_sz; i++)
        {
            sigma += (curr_mtx_ptr->get_weight(i, j) * in_lyr->get_node_value_idx(i));
                        
        }
        sigma += out_lyr->get_node_bias_idx(j);
        sigma /= in_lyr->get_num_nodes();
        sigma = m_pActFunc->apply_act_func(sigma);
        out_lyr->set_node_value(sigma, j);
        sigma = 0;
    }

    bRet = true;

    return bRet;
}

void NeuralNet::dump_nn()
{

    //todo corner conditions to be checked .... prone to crashes

    //setup and open dump file
    char fileName[MAX_DUMP_FILE_NAME_STR_SIZE];
    char dump_file_num_str[MAX_DUMP_FILE_NUM_STR_SIZE];

    dump_file_num_str[0] = (char)static_nDumpFileNum/10;
    dump_file_num_str[0] += '0';
    dump_file_num_str[1] = (char)static_nDumpFileNum%10;
    dump_file_num_str[1] += '0';
    dump_file_num_str[2] = '\0';

    strcpy(fileName, static_NNDumpFilePath);
    strcat(fileName, "nn");
    strcat(fileName, dump_file_num_str);
    strcat(fileName, ".dmp");

    
    if(!m_bInitialized)
    {
        return;
    }

    char * nn_str = new char[MAX_DUMP_FILE_SIZE];

    char temp[500];

    strcpy(nn_str, "NN Begin\n");

    //act func
    strcat(nn_str, "act func=");
    strcat(nn_str, static_const_parrActFuncStr[m_eActFunc]);
    strcat(nn_str, "\n");

    //number of layers
    strcat(nn_str, "num_lys=");
    sprintf(temp, "%d\n", m_unNumLys);
    strcat(nn_str, temp);

    

    //print layer node and bias values
    uint i = 0;
    uint j = 0;
    for(i = 0; i < m_unNumLys; i++)
    {
        sprintf(temp, "layer[%d] size=%d\n", i, m_ppLys[i]->get_num_nodes());
        strcat(nn_str, temp);
        for(j = 0; j < m_ppLys[i]->get_num_nodes(); j++)
        {
            if(m_ppLys[i]->get_layer_type() == INPUT_LYR)
            {
                sprintf(temp, "node[%d] : value=%f\n", j, m_ppLys[i]->get_node_value_idx(j));
                strcat(nn_str, temp);
            }
            else
            {
                sprintf(temp, "node[%d] : value=%f bias=%f\n", j, m_ppLys[i]->get_node_value_idx(j), m_ppLys[i]->get_node_bias_idx(j));
                strcat(nn_str, temp);
            }
        }

    }

    
    //print matrices
    for(i = 0; i < (m_unNumLys - 1); i++)
    {
        sprintf(temp, "matrix[%d]\n", i);
        strcat(nn_str, temp);
        for(j = 0; j < m_ppWtMtcs[i]->get_size(); j++)
        {
            uint x = j / m_ppLys[i + 1]->get_num_nodes();
            uint y = j % m_ppLys[i + 1]->get_num_nodes();
            sprintf(temp, "[%d][%d]%f ", x, y, m_ppWtMtcs[i]->get_weight(x, y));
            strcat(nn_str, temp);
        }
        strcat(nn_str, "\n");
    }

    strcat(nn_str, "NN END\n");

    FILE* dump_file;
    
    dump_file = fopen(fileName, "w");

    //write nn content into dump file

    fprintf(dump_file, "%s", nn_str);

    fclose(dump_file);

    delete [] nn_str;

    //increment dump_file_num for next dump
    static_nDumpFileNum++;

    

}

bool NeuralNet::populate_weights(uint unIdx, float* pfValues)
{
    bool bRet = false;

    if(!m_bInitialized)
    {
        return bRet;
    }

    if(unIdx >= m_unNumLys - 1)
    {
        return bRet;
    }

    bRet = m_ppWtMtcs[unIdx]->set_all_weight(pfValues);

    return bRet;

}

bool NeuralNet::populate_nodes_bias(uint unLyrIdx, float* pfBias)
{
    bool bRet = false;

    if(!m_bInitialized)
    {
        return bRet;
    }

    if(unLyrIdx >= m_unNumLys)
    {
        return bRet;
    }

    bRet = m_ppLys[unLyrIdx]->set_all_node_biases(pfBias);

    return bRet;
}

float NeuralNet::calculate_error(float* pfExpOut, float* pfError)
{
    float fRet = 0;
    if(!m_bInitialized)
    {
        return fRet;
    }

    nn_layer* output_lyr = m_ppLys[m_unNumLys - 1];

    uint i = 0;

    for(i = 0; i < output_lyr->get_num_nodes(); i++)
    {
        //todo: make generic to use other type of error functions
        //1/2 * squared error
        pfError[i] = pfExpOut[i] - output_lyr->get_node_value_idx(i);

        pfError[i] *= pfError[i];

        fRet += pfError[i];
    }

    fRet /= output_lyr->get_num_nodes();

    return fRet;

}

bool NeuralNet::do_backward_pass(float* pfExpOut)
{
    bool bRet = false;

    if(!m_bInitialized)
    {
        return bRet;
    }

    //calculate error for every layer's node. except input layer

    /*
        for output layer nodes:
            error = der_act_func(actual value) * (exp_value - actual_value)
        for hidden layer nodes:
            error = der_act_func(actual value) * (sum(weights_leading_out_of_node * error_of_node_it_is_reaching))
    */

    find_delta_of_all_nodes(pfExpOut);

    m_pOptimizer->correct_weights_biases();

    bRet = true;

    return bRet;

}

float NeuralNet::find_delta_of_all_nodes(float* pfExpOut)
{

    if(!m_bInitialized)
    {
        return -1.0;
    }

    uint i = 0; //layer index
    uint j = 0; //current layer node index
    uint k = 0; //previous layer node index

    float* error = new float[m_ppLys[m_unNumLys - 1]->get_num_nodes()];

    float total_error = calculate_error(pfExpOut, error);

    float temp = 0.0;

    // sigma += out_lyr->get_node_bias_idx(j);
    // sigma /= in_lyr->get_num_nodes();
    // NNLOG_DEBUG("layer [%d]: node[%d]:net=%f", in_layer_idx + 1, j, sigma);
    // sigma = apply_act_func(sigma);
    // NNLOG_DEBUG("layer [%d]: node[%d]=%f", in_layer_idx + 1, j, sigma);
    // out_lyr->set_node_value(sigma, j);

    for(i = (m_unNumLys - 1); i > INPUT_LAYER_ID; i--)
    {
        if(m_ppLys[i]->get_layer_type() == OUTPUT_LYR)//output layer
        {
            for(j = 0; j < m_ppLys[i]->get_num_nodes(); j++)
            {
                temp = -1.0 * (pfExpOut[j] - m_ppLys[i]->get_node_value_idx(j)); //deivative of error function
                //temp *= m_lys[i - 1]->get_num_nodes();
                temp *= m_pActFunc->apply_act_func_derv(m_ppLys[i]->get_node_value_idx(j));
                m_ppLys[i]->set_node_delta(temp, j);
                // m_ppLys[i]->set_node_bias((m_ppLys[i]->get_node_bias_idx(j) - (m_fLearningRate * temp)), j);
            }
        }
        else //hidden layer
        {
            for(j = 0; j < m_ppLys[i]->get_num_nodes(); j++)
            {
                temp = 0.0;

                for(k = 0; k < m_ppLys[i + 1]->get_num_nodes(); k++)
                {
                    temp += m_ppLys[i + 1]->get_node_delta_idx(k) * m_ppWtMtcs[i]->get_weight(j, k);
                }
                //temp *= m_lys[i - 1]->get_num_nodes();
                temp *= m_pActFunc->apply_act_func_derv(m_ppLys[i]->get_node_value_idx(j));
                m_ppLys[i]->set_node_delta(temp, j);
                // m_ppLys[i]->set_node_bias((m_ppLys[i]->get_node_bias_idx(j) - (m_fLearningRate * temp)), j);
            }
        }
        

    }

    delete [] error;

    return total_error;

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
    for(uint j = 0; j < m_unNumLys - 1 ; j++)
    {
        for(uint k = 0; k < m_ppWtMtcs[j]->get_size(); k++)
        {
            m_ppWtMtcs[j]->set_weight(k, m_ppWtMtcs[j]->get_weight(k) + m_batch_nns[i]->GetWeight(j, k));
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
       
    m_pOptimizer->correct_weights_biases();
    
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
        if(i != (m_unNumLys - 1))
        {
            m_ppWtMtcs[i]->populateWeightsWithRandomNumbers();
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
    return m_ppWtMtcs[MtxID]->get_size();
}

float NeuralNet::GetWeight(uint MtxID, uint Idx)
{
    return m_ppWtMtcs[MtxID]->get_weight(Idx);
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
    for(i = 0; i < (m_unNumLys - 1); i++)
    {
        sz = other->GetSzMtx(i);
        for(j = 0; j < sz; j++)
        {
            m_ppWtMtcs[i]->set_weight(j, other->GetWeight(i, j));
        }
    }
}

bool NeuralNet::isCorrectPrediction(float* pfOut)
{
    bool bRet = false;

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
    m_ppWtMtcs[MtxId]->set_weight(inIdx, outIdx, val);
}

float NeuralNet::GetWeight(uint MtxID, uint inIdx, uint outIdx)
{
    return m_ppWtMtcs[MtxID]->get_weight(inIdx, outIdx);
}

