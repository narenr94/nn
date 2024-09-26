#include "nn_core.h"

static const char* static_NNDumpFilePath = "./"; //dump file path

static int static_nDumpFileNum = 0; //number postfix fro dump files

/*
    list of activation functions
*/
static const char * static_const_parrActFuncStr[SIGMOID+1] =
{
	"SIGMOID" //Sigmoid
};

nn_node::nn_node()
{
    m_fValue = 0.0;
    m_fBias = 0.0;

    m_fDelta = 0.0;
}

void nn_node::set_bias(float fVal)
{
    NNLOG_TRACE("entering bias:%f", fVal);
    m_fBias = fVal;
    NNLOG_TRACE("exiting");
}

void nn_node::set_value(float fVal)
{
    NNLOG_TRACE("entering n:%f", fVal);    
    m_fValue = fVal;    
    NNLOG_TRACE("exiting");
}

void nn_node::set_delta(float fVal)
{
    NNLOG_TRACE("entering n:%f", fVal);    
    m_fDelta = fVal;    
    NNLOG_TRACE("exiting");
}

float nn_node::get_delta()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return m_fDelta;
}

float nn_node::get_bias()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return m_fBias;
}

float nn_node::get_value()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return m_fValue;    
}

uint nn_layer::get_num_nodes()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return m_unNumNodes;
    
}


nn_layer::~nn_layer()
{
    NNLOG_TRACE("entering");
    if(m_ppNodes)
    {
        free(m_ppNodes);
    }
    NNLOG_TRACE("exiting");
}

nn_layer::nn_layer(uint unNumNodesnodes)
{
    NNLOG_TRACE("entering n_nodes:%d", unNumNodesnodes);
    m_unNumNodes = unNumNodesnodes;

    m_ppNodes = (nn_node**)malloc(m_unNumNodes*sizeof(nn_node));

    uint i = 0;

    for(i = 0; i < m_unNumNodes; i++)
    {
        m_ppNodes[i] = new nn_node();
    }

    m_bInitialized = true;
    NNLOG_TRACE("exiting");
}

bool nn_layer::set_node_value(float fVal, uint unIdx)
{
    NNLOG_TRACE("entering val:%f idx:%d", fVal, unIdx);
    bool bRet = false;

    if(unIdx < m_unNumNodes && m_bInitialized)
    {
        m_ppNodes[unIdx]->set_value(fVal);

        bRet = true;
    }

    NNLOG_TRACE("exiting");

    return bRet;
}

bool nn_layer::set_all_node_values(float* pfValue)
{
    NNLOG_TRACE("entering");
    bool bRet = false;

    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting not initialized!!");
        return bRet;
    }

    bRet = true;

    uint i = 0;

    for(i = 0; i < m_unNumNodes; i++)
    {
        m_ppNodes[i]->set_value(pfValue[i]);
        //NNLOG_MIL("i:%d node[i]->get_value():%f value[i]:%f", i, nodes[i]->get_value(), value[i]);
    }

    NNLOG_TRACE("exiting");

    return bRet;
}

bool nn_layer::set_all_node_biases(float* pfBias)
{
    NNLOG_TRACE("entering");
    bool bRet = false;

    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting not initialized!!!");
        return bRet;
    }

    bRet = true;

    uint i = 0;

    for(i = 0; i < m_unNumNodes; i++)
    {
        m_ppNodes[i]->set_bias(pfBias[i]);
    }

    NNLOG_TRACE("exiting");

    return bRet;

}

bool nn_layer::set_node_bias(float fBias, uint unIdx)
{
    NNLOG_TRACE("entering bias:%f idx:%d", fBias, unIdx);
    bool bRet = false;

    if((unIdx < m_unNumNodes) && m_bInitialized)
    {
        m_ppNodes[unIdx]->set_bias(fBias);

        bRet = true;
    }

    NNLOG_TRACE("exiting");

    return bRet;
}

bool nn_layer::set_node_delta(float fDelta, uint unIdx)
{
    NNLOG_TRACE("entering delta:%f idx:%d", fDelta, unIdx);
    bool bRet = false;

    if((unIdx < m_unNumNodes) && m_bInitialized)
    {
        m_ppNodes[unIdx]->set_delta(fDelta);

        bRet = true;
    }

    NNLOG_TRACE("exiting");

    return bRet;
}

bool nn_layer::is_initialized()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return m_bInitialized;
}

float nn_layer::get_node_value_idx(uint unIdx)
{
    NNLOG_TRACE("entering idx:%d", unIdx);
    NNLOG_TRACE("exiting");
    return m_ppNodes[unIdx]->get_value();
}

float nn_layer::get_node_bias_idx(uint unIdx)
{
    NNLOG_TRACE("entering idx:%d", unIdx);
    NNLOG_TRACE("exiting");
    return m_ppNodes[unIdx]->get_bias();
}

float nn_layer::get_node_delta_idx(uint unIdx)
{
    NNLOG_TRACE("entering idx:%d", unIdx);
    NNLOG_TRACE("exiting");
    return m_ppNodes[unIdx]->get_delta();
}

void nn_layer::set_layer_type(eLyr_type eLType)
{
    NNLOG_TRACE("entering lt:%d", eLType);
    eLyrType = eLType;
    NNLOG_TRACE("exiting");
}

eLyr_type nn_layer::get_layer_type()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return eLyrType;

}

void nn_layer::populateBiasesWithRandomNumbers()
{
    NNLOG_TRACE("entering");

    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting not initialized!!!");
        return;
    }

    uint i = 0;
    float tmp;

    for(i = 0; i < m_unNumNodes; i++)
    {
        tmp = (float)getRandomNumber(RAND_MIN_WEIGHT_BIAS, RAND_MAX_WEIGHT_BIAS);
        tmp /= 10.0;
        m_ppNodes[i]->set_bias(tmp);
    }

    NNLOG_TRACE("exiting");

}

nn_l2l_weight_matrix::nn_l2l_weight_matrix(nn_layer* pInL, nn_layer* pOutL)
{
    NNLOG_TRACE("entering");

    if(pInL->is_initialized() && pOutL->is_initialized())
    {
        m_pInputLayer = pInL;
        m_pOutputLayer = pOutL;

        uint in_l_size = m_pInputLayer->get_num_nodes();
        uint out_l_size = m_pOutputLayer->get_num_nodes();

        m_unSize = in_l_size * out_l_size;

        m_pfWeightMatrix = (float*)malloc(m_unSize * sizeof(float));

        m_bInitialized = true;

    }

    NNLOG_TRACE("exiting");
}


nn_l2l_weight_matrix::~nn_l2l_weight_matrix()
{
    NNLOG_TRACE("entering");
    if(m_pfWeightMatrix)
    {
        free(m_pfWeightMatrix);
    }
    NNLOG_TRACE("exiting");
}


bool nn_l2l_weight_matrix::set_weight(uint unInIdx, uint unOutIdx, float fWt)
{
    NNLOG_TRACE("entering in_idx:%d out_idx:%d wt:%f", unInIdx, unOutIdx, fWt);

    bool bRet = false;

    if(m_bInitialized)
    {
        if((unInIdx < m_pInputLayer->get_num_nodes()) && (unOutIdx < m_pOutputLayer->get_num_nodes()))
        {
            m_pfWeightMatrix[(unInIdx * m_pOutputLayer->get_num_nodes()) + unOutIdx] = fWt;

            bRet = true;
        }

    }
    else
    {
        NNLOG_ERR("exiting not initialized!!!");
    }

    NNLOG_TRACE("exiting");

    return bRet;

}

bool nn_l2l_weight_matrix::set_all_weight(float* fWt)
{
    NNLOG_TRACE("entering");
    bool bRet = false;

    if(m_bInitialized)
    {
        uint i = 0;
        uint j = 0;
        for(i = 0; i < m_pInputLayer->get_num_nodes(); i++)
        {
            for(j = 0; j < m_pOutputLayer->get_num_nodes(); j++)
            {
                m_pfWeightMatrix[(i * m_pOutputLayer->get_num_nodes()) + j] = fWt[(i * m_pOutputLayer->get_num_nodes()) + j];
            }
        }

        bRet = true;

    }
    else
    {
        NNLOG_ERR("exiting not initialized!!!");
    }

    NNLOG_TRACE("exiting");
    return bRet;
}

float nn_l2l_weight_matrix::get_weight(uint unInIdx, uint unOutIdx)
{
    NNLOG_TRACE("entering in_idx:%d out_idx:%d", unInIdx, unOutIdx);
    float bRet = 0;
    if(m_bInitialized)
    {
        bRet = m_pfWeightMatrix[(unInIdx * m_pOutputLayer->get_num_nodes()) + unOutIdx];
    }
    else
    {
        NNLOG_ERR("exiting not initialized!!!");
    }

    NNLOG_TRACE("exiting");

    return bRet;

}

uint nn_l2l_weight_matrix::get_size()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return m_unSize;
}

void nn_l2l_weight_matrix::populateWeightsWithRandomNumbers()
{
    NNLOG_TRACE("entering");
    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting not initialized!!!");
        return;
    }

    uint i = 0;

    for(i = 0; i < m_unSize; i++)
    {
        m_pfWeightMatrix[i] = (float)getRandomNumber(RAND_MIN_WEIGHT_BIAS, RAND_MAX_WEIGHT_BIAS);
        m_pfWeightMatrix[i] /= 10.0;
    }

    NNLOG_TRACE("exiting");
}


NeuralNet::NeuralNet()
{
    NNLOG_TRACE("entering");
    m_eActFunc = SIGMOID;
    m_ppWtMtcs = NULL;
    m_ppLys = NULL;

    m_bInitialized = false;
    NNLOG_TRACE("exiting");
}

NeuralNet::~NeuralNet()
{
    NNLOG_TRACE("entering");
    if(m_ppWtMtcs)
    {
        free(m_ppWtMtcs);
    }
    if(m_ppLys)
    {
        free(m_ppLys);
    }
    NNLOG_TRACE("exiting");
}

bool NeuralNet::init(uint unNoLys, uint* unpSzLys, eAct_func eAFunc, elog_level eLLevel, bool bConsolePrint, float fLRate)
{
    init_nn_logger(eLLevel, bConsolePrint);

    NNLOG_TRACE("entering no_lys:%d act_func:%d loglevel:%d consolePrint:%d", unNoLys, eAFunc, eLLevel, bConsolePrint);
    
    bool bRet = false;
    //minimum layers is 3
    if(unNoLys < 3)
    {
        return bRet;
    }

    if(fLRate > 1.0 || fLRate <= 0.0)
    {
        return bRet;
    }

    srand(time(NULL));

    m_eActFunc = eAFunc;

    m_fLearningRate = fLRate;

    uint i = 0; //local var for iterating loops

    m_unNumLys = unNoLys;

    //allocate layers
    m_ppLys = (nn_layer**)malloc(m_unNumLys * sizeof(nn_layer*));

    for(i = 0; i < m_unNumLys; i++)
    {
        m_ppLys[i] = new nn_layer(unpSzLys[i]);
        if(i == INPUT_LAYER_ID)
        {
            m_ppLys[i]->set_layer_type(INPUT_LYR);
        }
        else if(i == (m_unNumLys -1))
        {
            m_ppLys[i]->set_layer_type(OUTPUT_LYR);
        }
        else
        {
            m_ppLys[i]->set_layer_type(HIDDEN_LYR);
        }
    }
    
    //allocate weight matrices
    m_ppWtMtcs = (nn_l2l_weight_matrix**)malloc((m_unNumLys - 1) * sizeof(nn_l2l_weight_matrix*));

    for(i = 0; i < (m_unNumLys - 1); i++)
    {
        m_ppWtMtcs[i] = new nn_l2l_weight_matrix(m_ppLys[i], m_ppLys[i + 1]);
        
    }
    
    m_bInitialized = true;

    bRet = true;

    NNLOG_TRACE("exiting");

    return bRet;
    

}

bool NeuralNet::do_forward_pass(float* pfInputArr)
{

    NNLOG_TRACE("entering");

    NNLOG_MIL("Forward Propogation Started");

    bool bRet = false;

    //check if NN is initalised
    if(!m_bInitialized)
    {
        NNLOG_ERR("Not Initialized!!!");
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

    NNLOG_TRACE("exiting");

    NNLOG_MIL("Forward Propogation Complete");

    return bRet;

}

bool NeuralNet::do_forwardpass_to_next_layer(uint unInLayerIdx)
{
    NNLOG_TRACE("entering in_layer_idx:%d", unInLayerIdx);
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
            NNLOG_DEBUG("layer [%d]: wt[%d][%d]=%f in_lyr_val=%f", unInLayerIdx + 1, i, j, curr_mtx_ptr->get_weight(i, j), in_lyr->get_node_value_idx(i));
            // if(unInLayerIdx == INPUT_LAYER_ID)
            // {
            //     NNLOG_MIL("layer [%d]: wt[%d][%d]=%f in_lyr_val=%f", unInLayerIdx + 1, i, j, curr_mtx_ptr->get_weight(i, j), in_lyr->get_node_value_idx(i));
            // }
            
        }
        sigma += out_lyr->get_node_bias_idx(j);
        sigma /= in_lyr->get_num_nodes();
        NNLOG_DEBUG("layer [%d]: node[%d]:net=%f", unInLayerIdx + 1, j, sigma);
        sigma = apply_act_func(sigma);
        NNLOG_DEBUG("layer [%d]: node[%d]=%f", unInLayerIdx + 1, j, sigma);
        out_lyr->set_node_value(sigma, j);
        sigma = 0;
    }

    bRet = true;

    NNLOG_TRACE("exiting");

    return bRet;
}

void NeuralNet::dump_nn()
{

    NNLOG_TRACE("entering");

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
        NNLOG_ERR("exiting, not initialized!!!");
        return;
    }

    char * nn_str = (char *)malloc(MAX_DUMP_FILE_SIZE * sizeof(char));

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
    
    NNLOG_DEBUG("fileName: %s", fileName);

    NNLOG_DEBUG("dump content \n%s\ndone", nn_str);


    dump_file = fopen(fileName, "w");

    //write nn content into dump file

    fprintf(dump_file, "%s", nn_str);

    fclose(dump_file);

    free(nn_str);

    //increment dump_file_num for next dump
    static_nDumpFileNum++;

    NNLOG_TRACE("exiting");


}

bool NeuralNet::populate_weights(uint unIdx, float* pfValues)
{
    NNLOG_TRACE("entering idx:%d", unIdx);
    bool bRet = false;

    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return bRet;
    }

    if(unIdx >= m_unNumLys - 1)
    {
        NNLOG_ERR("exiting, idx >= num_lys - 1 !!!");
        return bRet;
    }

    bRet = m_ppWtMtcs[unIdx]->set_all_weight(pfValues);

    NNLOG_TRACE("exiting");

    return bRet;

}

bool NeuralNet::populate_nodes_bias(uint unLyrIdx, float* pfBias)
{
    NNLOG_TRACE("entering lyr_idx:%d", unLyrIdx);
    bool bRet = false;

    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return bRet;
    }

    if(unLyrIdx >= m_unNumLys)
    {
        NNLOG_ERR("exiting, idx >= num_lys!!!");
        return bRet;
    }

    bRet = m_ppLys[unLyrIdx]->set_all_node_biases(pfBias);

    NNLOG_TRACE("exiting");
    

    return bRet;
}

float NeuralNet::calculate_error(float* pfExpOut, float* pfError)
{
    float fRet = 0;
    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
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

        NNLOG_DEBUG("node[%d] pfExpOut=%f actual_out=%f error=%f", i, pfExpOut[i], output_lyr->get_node_value_idx(i), pfError[i]);
        //NNLOG_MIL("node[%d] pfExpOut=%f actual_out=%f", i, pfExpOut[i], output_lyr->get_node_value_idx(i));
        
        fRet += pfError[i];
    }

    fRet /= output_lyr->get_num_nodes();

    NNLOG_MIL("total error=%f", fRet);

    return fRet;

}

bool NeuralNet::do_backward_pass(float* pfExpOut)
{
    bool bRet = false;

    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return bRet;
    }

    NNLOG_MIL("Backward Propogation Started");

    //calculate error for every layer's node. except input layer

    /*
        for output layer nodes:
            error = der_act_func(actual value) * (exp_value - actual_value)
        for hidden layer nodes:
            error = der_act_func(actual value) * (sum(weights_leading_out_of_node * error_of_node_it_is_reaching))
    */

    find_delta_of_all_nodes_and_correct_biases(pfExpOut);

    correct_weights();

    bRet = true;

    NNLOG_MIL("Backward Propogation Complete");

    return bRet;

}

float NeuralNet::apply_act_func(float n)
{
    float fRet = 0.0;
    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return fRet;
    }

    switch(m_eActFunc)
    {
        case SIGMOID:
            fRet = get_sigmoidf(n);

        default:
            break;
    }

    return fRet;
}

float NeuralNet::apply_act_func_derv(float fVal)
{
    float fRet = 0.0;
    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return fRet;
    }

    switch(m_eActFunc)
    {
        case SIGMOID:
            fRet = find_derivative_sigmoidf(fVal);

        default:
            break;
    }

    return fRet;
}

float NeuralNet::find_delta_of_all_nodes_and_correct_biases(float* pfExpOut)
{

    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return -1.0;
    }

    uint i = 0; //layer index
    uint j = 0; //current layer node index
    uint k = 0; //previous layer node index

    float* error = (float*)malloc(m_ppLys[m_unNumLys - 1]->get_num_nodes() * sizeof(float));

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
                NNLOG_DEBUG("-(target_Node - act_node)=%f", temp);
                temp *= apply_act_func_derv(m_ppLys[i]->get_node_value_idx(j));
                NNLOG_DEBUG("apply_act_func_derv=%f val:%f", apply_act_func_derv(m_ppLys[i]->get_node_value_idx(j)), m_ppLys[i]->get_node_value_idx(j));
                m_ppLys[i]->set_node_delta(temp, j);
                NNLOG_DEBUG("layer[%d] node[%d] bias=%f delta=%f learningRate=%f", i, j, m_ppLys[i]->get_node_bias_idx(j), temp, m_fLearningRate);
                m_ppLys[i]->set_node_bias((m_ppLys[i]->get_node_bias_idx(j) - (m_fLearningRate * temp)), j);
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
                    NNLOG_DEBUG("prev layer[%d] node[%d] deltas:%f wt:%f", i + 1, k,m_ppLys[i + 1]->get_node_delta_idx(k), m_ppWtMtcs[i]->get_weight(j, k));
                }
                //temp *= m_lys[i - 1]->get_num_nodes();
                NNLOG_DEBUG("sum of prev layer nodes deltas * appropriate wts:%f node_val:%f", temp, m_ppLys[i]->get_node_value_idx(j));
                temp *= apply_act_func_derv(m_ppLys[i]->get_node_value_idx(j));
                m_ppLys[i]->set_node_delta(temp, j);
                NNLOG_DEBUG("layer[%d] node[%d] bias=%f delta=%f learningRate=%f apply_act_func_derv:%f", i, j, m_ppLys[i]->get_node_bias_idx(j), temp, m_fLearningRate, apply_act_func_derv(m_ppLys[i]->get_node_value_idx(j)));
                m_ppLys[i]->set_node_bias((m_ppLys[i]->get_node_bias_idx(j) - (m_fLearningRate * temp)), j);
            }
        }
        

    }

    free(error);

    return total_error;

}

void NeuralNet::correct_weights()
{
    if(!m_bInitialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return;
    }

    
    uint i; //in layer index, out layer index is always in layer index + 1
    uint j; //in layer node index
    uint k; //out layer node index

    float delta_wt = 0.0;

    for(i = INPUT_LAYER_ID; i < m_unNumLys - 1; i++)
    {
        for(j = 0; j < m_ppLys[i]->get_num_nodes(); j++)
        {
            for(k = 0; k < m_ppLys[i + 1]->get_num_nodes(); k++)
            {
                delta_wt = m_ppLys[i + 1]->get_node_delta_idx(k) * m_ppLys[i]->get_node_value_idx(j);
                NNLOG_DEBUG("layer[%d] wt[%d][%d] out_node_delta=%f in_node_val=%f learningRate=%f old_wt:%f", i, j, k,  m_ppLys[i + 1]->get_node_delta_idx(k), m_ppLys[i]->get_node_value_idx(j), m_fLearningRate, m_ppWtMtcs[i]->get_weight(j, k));
                delta_wt *= m_fLearningRate;

                m_ppWtMtcs[i]->set_weight(j, k, (m_ppWtMtcs[i]->get_weight(j, k) - delta_wt));

                NNLOG_DEBUG("layer[%d] wt[%d][%d] new_wt=%f delta_wt=%f", i, j, k,  m_ppWtMtcs[i]->get_weight(j, k), delta_wt);
            }
            

        }
    }    

}

bool NeuralNet::Train(float* in, float* out)
{
    bool bRet = false;

    do_forward_pass(in);

    bRet = isCorrectPrediction(out);

    do_backward_pass(out);

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

