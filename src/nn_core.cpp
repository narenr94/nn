#include "nn_core.h"

static const char* nn_dump_file_path = "./"; //dump file path

static int dump_file_num = 0; //number postfix fro dump files

/*
    list of activation functions
*/
static const char *m_eAct_funcStr[SIGMOID+1] =
{
	"SIGMOID" //Sigmoid
};

nn_node::nn_node()
{
    value = 0.0;
    bias = 0.0;

    delta = 0.0;
}

void nn_node::set_bias(float n)
{
    NNLOG_TRACE("entering bias:%f", n);
    bias = n;
    NNLOG_TRACE("exiting");
}

void nn_node::set_value(float n)
{
    NNLOG_TRACE("entering n:%f", n);    
    value = n;    
    NNLOG_TRACE("exiting");
}

void nn_node::set_delta(float n)
{
    NNLOG_TRACE("entering n:%f", n);    
    delta = n;    
    NNLOG_TRACE("exiting");
}

float nn_node::get_delta()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return delta;
}

float nn_node::get_bias()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return bias;
}

float nn_node::get_value()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return value;    
}

uint nn_layer::get_num_nodes()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return num_nodes;
    
}


nn_layer::~nn_layer()
{
    NNLOG_TRACE("entering");
    if(nodes)
    {
        free(nodes);
    }
    NNLOG_TRACE("exiting");
}

nn_layer::nn_layer(uint n_nodes)
{
    NNLOG_TRACE("entering n_nodes:%d", n_nodes);
    num_nodes = n_nodes;

    nodes = (nn_node**)malloc(num_nodes*sizeof(nn_node));

    uint i = 0;

    for(i = 0; i < num_nodes; i++)
    {
        nodes[i] = new nn_node();
    }

    initialized = true;
    NNLOG_TRACE("exiting");
}

bool nn_layer::set_node_value(float val, uint idx)
{
    NNLOG_TRACE("entering val:%f idx:%d", val, idx);
    bool ret = false;

    if(idx < num_nodes && initialized)
    {
        nodes[idx]->set_value(val);

        ret = true;
    }

    NNLOG_TRACE("exiting");

    return ret;
}

bool nn_layer::set_all_node_values(float* value)
{
    NNLOG_TRACE("entering");
    bool ret = false;

    if(!initialized)
    {
        NNLOG_ERR("exiting not initialized!!");
        return ret;
    }

    ret = true;

    uint i = 0;

    for(i = 0; i < num_nodes; i++)
    {
        nodes[i]->set_value(value[i]);
        //NNLOG_MIL("i:%d node[i]->get_value():%f value[i]:%f", i, nodes[i]->get_value(), value[i]);
    }

    NNLOG_TRACE("exiting");

    return ret;
}

bool nn_layer::set_all_node_biases(float* bias)
{
    NNLOG_TRACE("entering");
    bool ret = false;

    if(!initialized)
    {
        NNLOG_ERR("exiting not initialized!!!");
        return ret;
    }

    ret = true;

    uint i = 0;

    for(i = 0; i < num_nodes; i++)
    {
        nodes[i]->set_bias(bias[i]);
    }

    NNLOG_TRACE("exiting");

    return ret;

}

bool nn_layer::set_node_bias(float b, uint idx)
{
    NNLOG_TRACE("entering bias:%f idx:%d", b, idx);
    bool ret = false;

    if((idx < num_nodes) && initialized)
    {
        nodes[idx]->set_bias(b);

        ret = true;
    }

    NNLOG_TRACE("exiting");

    return ret;
}

bool nn_layer::set_node_delta(float delta, uint idx)
{
    NNLOG_TRACE("entering delta:%f idx:%d", delta, idx);
    bool ret = false;

    if((idx < num_nodes) && initialized)
    {
        nodes[idx]->set_delta(delta);

        ret = true;
    }

    NNLOG_TRACE("exiting");

    return ret;
}

bool nn_layer::is_initialized()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return initialized;
}

float nn_layer::get_node_value_idx(uint idx)
{
    NNLOG_TRACE("entering idx:%d", idx);
    NNLOG_TRACE("exiting");
    return nodes[idx]->get_value();
}

float nn_layer::get_node_bias_idx(uint idx)
{
    NNLOG_TRACE("entering idx:%d", idx);
    NNLOG_TRACE("exiting");
    return nodes[idx]->get_bias();
}

float nn_layer::get_node_delta_idx(uint idx)
{
    NNLOG_TRACE("entering idx:%d", idx);
    NNLOG_TRACE("exiting");
    return nodes[idx]->get_delta();
}

void nn_layer::set_layer_type(eLyr_type lt)
{
    NNLOG_TRACE("entering lt:%d", lt);
    lyrType = lt;
    NNLOG_TRACE("exiting");
}

eLyr_type nn_layer::get_layer_type()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return lyrType;

}

void nn_layer::populateBiasesWithRandomNumbers()
{
    NNLOG_TRACE("entering");

    if(!initialized)
    {
        NNLOG_ERR("exiting not initialized!!!");
        return;
    }

    uint i = 0;
    float tmp;

    for(i = 0; i < num_nodes; i++)
    {
        tmp = (float)getRandomNumber(RAND_MIN_WEIGHT_BIAS, RAND_MAX_WEIGHT_BIAS);
        tmp /= 10.0;
        nodes[i]->set_bias(tmp);
    }

    NNLOG_TRACE("exiting");

}

nn_l2l_weight_matrix::nn_l2l_weight_matrix(nn_layer* in_l, nn_layer* out_l)
{
    NNLOG_TRACE("entering");

    if(in_l->is_initialized() && out_l->is_initialized())
    {
        input_layer = in_l;
        output_layer = out_l;

        uint in_l_size = input_layer->get_num_nodes();
        uint out_l_size = output_layer->get_num_nodes();

        size = in_l_size * out_l_size;

        weight_matrix = (float*)malloc(size * sizeof(float));

        initialized = true;

    }

    NNLOG_TRACE("exiting");
}


nn_l2l_weight_matrix::~nn_l2l_weight_matrix()
{
    NNLOG_TRACE("entering");
    if(weight_matrix)
    {
        free(weight_matrix);
    }
    NNLOG_TRACE("exiting");
}


bool nn_l2l_weight_matrix::set_weight(uint in_idx, uint out_idx, float wt)
{
    NNLOG_TRACE("entering in_idx:%d out_idx:%d wt:%f", in_idx, out_idx, wt);

    bool ret = false;

    if(initialized)
    {
        if((in_idx < input_layer->get_num_nodes()) && (out_idx < output_layer->get_num_nodes()))
        {
            weight_matrix[(in_idx * output_layer->get_num_nodes()) + out_idx] = wt;

            ret = true;
        }

    }
    else
    {
        NNLOG_ERR("exiting not initialized!!!");
    }

    NNLOG_TRACE("exiting");

    return ret;

}

bool nn_l2l_weight_matrix::set_all_weight(float* wt)
{
    NNLOG_TRACE("entering");
    bool ret = false;

    if(initialized)
    {
        uint i = 0;
        uint j = 0;
        for(i = 0; i < input_layer->get_num_nodes(); i++)
        {
            for(j = 0; j < output_layer->get_num_nodes(); j++)
            {
                weight_matrix[(i * output_layer->get_num_nodes()) + j] = wt[(i * output_layer->get_num_nodes()) + j];
            }
        }

        ret = true;

    }
    else
    {
        NNLOG_ERR("exiting not initialized!!!");
    }

    NNLOG_TRACE("exiting");
    return ret;
}

float nn_l2l_weight_matrix::get_weight(uint in_idx, uint out_idx)
{
    NNLOG_TRACE("entering in_idx:%d out_idx:%d", in_idx, out_idx);
    float ret = 0;
    if(initialized)
    {
        ret = weight_matrix[(in_idx * output_layer->get_num_nodes()) + out_idx];
    }
    else
    {
        NNLOG_ERR("exiting not initialized!!!");
    }

    NNLOG_TRACE("exiting");

    return ret;

}

uint nn_l2l_weight_matrix::get_size()
{
    NNLOG_TRACE("entering");
    NNLOG_TRACE("exiting");
    return size;
}

void nn_l2l_weight_matrix::populateWeightsWithRandomNumbers()
{
    NNLOG_TRACE("entering");
    if(!initialized)
    {
        NNLOG_ERR("exiting not initialized!!!");
        return;
    }

    uint i = 0;

    for(i = 0; i < size; i++)
    {
        weight_matrix[i] = (float)getRandomNumber(RAND_MIN_WEIGHT_BIAS, RAND_MAX_WEIGHT_BIAS);
        weight_matrix[i] /= 10.0;
    }

    NNLOG_TRACE("exiting");
}


NeuralNet::NeuralNet()
{
    NNLOG_TRACE("entering");
    m_act_func = SIGMOID;
    m_wt_mtcs = NULL;
    m_lys = NULL;

    initialized = false;
    NNLOG_TRACE("exiting");
}

NeuralNet::~NeuralNet()
{
    NNLOG_TRACE("entering");
    if(m_wt_mtcs)
    {
        free(m_wt_mtcs);
    }
    if(m_lys)
    {
        free(m_lys);
    }
    NNLOG_TRACE("exiting");
}

bool NeuralNet::init(uint no_lys, uint* sz_lys, eAct_func act_func, elog_level loglevel, bool consolePrint, float lRate)
{
    init_nn_logger(loglevel, consolePrint);

    NNLOG_TRACE("entering no_lys:%d act_func:%d loglevel:%d consolePrint:%d", no_lys, act_func, loglevel, consolePrint);
    
    bool ret = false;
    //minimum layers is 3
    if(no_lys < 3)
    {
        return ret;
    }

    if(lRate > 1.0 || lRate <= 0.0)
    {
        return ret;
    }

    srand(time(NULL));

    m_act_func = act_func;

    learningRate = lRate;

    uint i = 0; //local var for iterating loops

    num_lys = no_lys;

    //allocate layers
    m_lys = (nn_layer**)malloc(no_lys * sizeof(nn_layer*));

    for(i = 0; i < no_lys; i++)
    {
        m_lys[i] = new nn_layer(sz_lys[i]);
        if(i == INPUT_LAYER_ID)
        {
            m_lys[i]->set_layer_type(INPUT_LYR);
        }
        else if(i == (no_lys -1))
        {
            m_lys[i]->set_layer_type(OUTPUT_LYR);
        }
        else
        {
            m_lys[i]->set_layer_type(HIDDEN_LYR);
        }
    }
    
    //allocate weight matrices
    m_wt_mtcs = (nn_l2l_weight_matrix**)malloc((no_lys - 1) * sizeof(nn_l2l_weight_matrix*));

    for(i = 0; i < (no_lys - 1); i++)
    {
        m_wt_mtcs[i] = new nn_l2l_weight_matrix(m_lys[i], m_lys[i + 1]);
        
    }
    
    initialized = true;

    ret = true;

    NNLOG_TRACE("exiting");

    return ret;
    

}

bool NeuralNet::forward_propagation(float* input_arr)
{

    NNLOG_TRACE("entering");

    NNLOG_MIL("Forward Propogation Started");

    bool ret = false;

    //check if NN is initalised
    if(!initialized)
    {
        NNLOG_ERR("Not Initialized!!!");
        return ret;
    }

    //set input layer nodes
    m_lys[INPUT_LAYER_ID]->set_all_node_values(input_arr);

    uint i = 0;

    // for(i = 0; i < 784; i++)
    // {
    //     NNLOG_MIL("[%d]%f", i, m_lys[INPUT_LAYER_ID]->get_node_value_idx(i));
    // }

    for(i = 0; i < (num_lys - 1); i++)
    {
        if(!forwardpass_to_next_layer(INPUT_LAYER_ID + i))
        {
            return ret;
        }
    }
    
    ret = true;

    NNLOG_TRACE("exiting");

    NNLOG_MIL("Forward Propogation Complete");

    return ret;

}

bool NeuralNet::forwardpass_to_next_layer(uint in_layer_idx)
{
    NNLOG_TRACE("entering in_layer_idx:%d", in_layer_idx);
    bool ret = false;

    if(!initialized)
    {
        return ret;
    }

    nn_layer* in_lyr = m_lys[in_layer_idx];
    nn_layer* out_lyr = m_lys[in_layer_idx + 1];

    nn_l2l_weight_matrix* curr_mtx_ptr = m_wt_mtcs[in_layer_idx];

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
            NNLOG_DEBUG("layer [%d]: wt[%d][%d]=%f in_lyr_val=%f", in_layer_idx + 1, i, j, curr_mtx_ptr->get_weight(i, j), in_lyr->get_node_value_idx(i));
            // if(in_layer_idx == INPUT_LAYER_ID)
            // {
            //     NNLOG_MIL("layer [%d]: wt[%d][%d]=%f in_lyr_val=%f", in_layer_idx + 1, i, j, curr_mtx_ptr->get_weight(i, j), in_lyr->get_node_value_idx(i));
            // }
            
        }
        sigma += out_lyr->get_node_bias_idx(j);
        sigma /= in_lyr->get_num_nodes();
        NNLOG_DEBUG("layer [%d]: node[%d]:net=%f", in_layer_idx + 1, j, sigma);
        sigma = apply_act_func(sigma);
        NNLOG_DEBUG("layer [%d]: node[%d]=%f", in_layer_idx + 1, j, sigma);
        out_lyr->set_node_value(sigma, j);
        sigma = 0;
    }

    ret = true;

    NNLOG_TRACE("exiting");

    return ret;
}

void NeuralNet::dump_nn()
{

    NNLOG_TRACE("entering");

    //todo corner conditions to be checked .... prone to crashes

    //setup and open dump file
    char fileName[MAX_DUMP_FILE_NAME_STR_SIZE];
    char dump_file_num_str[MAX_DUMP_FILE_NUM_STR_SIZE];

    dump_file_num_str[0] = (char)dump_file_num/10;
    dump_file_num_str[0] += '0';
    dump_file_num_str[1] = (char)dump_file_num%10;
    dump_file_num_str[1] += '0';
    dump_file_num_str[2] = '\0';

    strcpy(fileName, nn_dump_file_path);
    strcat(fileName, "nn");
    strcat(fileName, dump_file_num_str);
    strcat(fileName, ".dmp");

    
    if(!initialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return;
    }

    char * nn_str = (char *)malloc(MAX_DUMP_FILE_SIZE * sizeof(char));

    char temp[500];

    strcpy(nn_str, "NN Begin\n");

    //act func
    strcat(nn_str, "act func=");
    strcat(nn_str, m_eAct_funcStr[m_act_func]);
    strcat(nn_str, "\n");

    //number of layers
    strcat(nn_str, "num_lys=");
    sprintf(temp, "%d\n", num_lys);
    strcat(nn_str, temp);

    

    //print layer node and bias values
    uint i = 0;
    uint j = 0;
    for(i = 0; i < num_lys; i++)
    {
        sprintf(temp, "layer[%d] size=%d\n", i, m_lys[i]->get_num_nodes());
        strcat(nn_str, temp);
        for(j = 0; j < m_lys[i]->get_num_nodes(); j++)
        {
            if(m_lys[i]->get_layer_type() == INPUT_LYR)
            {
                sprintf(temp, "node[%d] : value=%f\n", j, m_lys[i]->get_node_value_idx(j));
                strcat(nn_str, temp);
            }
            else
            {
                sprintf(temp, "node[%d] : value=%f bias=%f\n", j, m_lys[i]->get_node_value_idx(j), m_lys[i]->get_node_bias_idx(j));
                strcat(nn_str, temp);
            }
        }

    }

    
    //print matrices
    for(i = 0; i < (num_lys - 1); i++)
    {
        sprintf(temp, "matrix[%d]\n", i);
        strcat(nn_str, temp);
        for(j = 0; j < m_wt_mtcs[i]->get_size(); j++)
        {
            uint x = j / m_lys[i + 1]->get_num_nodes();
            uint y = j % m_lys[i + 1]->get_num_nodes();
            sprintf(temp, "[%d][%d]%f ", x, y, m_wt_mtcs[i]->get_weight(x, y));
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
    dump_file_num++;

    NNLOG_TRACE("exiting");


}

bool NeuralNet::populate_weights(uint idx, float * values)
{
    NNLOG_TRACE("entering idx:%d", idx);
    bool ret = false;

    if(!initialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return ret;
    }

    if(idx >= num_lys - 1)
    {
        NNLOG_ERR("exiting, idx >= num_lys - 1 !!!");
        return ret;
    }

    ret = m_wt_mtcs[idx]->set_all_weight(values);

    NNLOG_TRACE("exiting");

    return ret;

}

bool NeuralNet::populate_nodes_bias(uint lyr_idx, float* bias)
{
    NNLOG_TRACE("entering lyr_idx:%d", lyr_idx);
    bool ret = false;

    if(!initialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return ret;
    }

    if(lyr_idx >= num_lys)
    {
        NNLOG_ERR("exiting, idx >= num_lys!!!");
        return ret;
    }

    ret = m_lys[lyr_idx]->set_all_node_biases(bias);

    NNLOG_TRACE("exiting");
    

    return ret;
}

float NeuralNet::calculate_error(float* exp_out, float* error)
{
    float ret = 0;
    if(!initialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return ret;
    }

    nn_layer* output_lyr = m_lys[num_lys - 1];

    uint i = 0;

    for(i = 0; i < output_lyr->get_num_nodes(); i++)
    {
        //todo: make generic to use other type of error functions
        //1/2 * squared error
        error[i] = exp_out[i] - output_lyr->get_node_value_idx(i);

        error[i] *= error[i];

        NNLOG_DEBUG("node[%d] exp_out=%f actual_out=%f error=%f", i, exp_out[i], output_lyr->get_node_value_idx(i), error[i]);
        //NNLOG_MIL("node[%d] exp_out=%f actual_out=%f", i, exp_out[i], output_lyr->get_node_value_idx(i));
        
        ret += error[i];
    }

    ret /= output_lyr->get_num_nodes();

    NNLOG_MIL("total error=%f", ret);

    return ret;

}

bool NeuralNet::backward_propogation(float* exp_out)
{
    bool ret = false;

    if(!initialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return ret;
    }

    NNLOG_MIL("Backward Propogation Started");

    //calculate error for every layer's node. except input layer

    /*
        for output layer nodes:
            error = der_act_func(actual value) * (exp_value - actual_value)
        for hidden layer nodes:
            error = der_act_func(actual value) * (sum(weights_leading_out_of_node * error_of_node_it_is_reaching))
    */

    find_delta_of_all_nodes_and_correct_biases(exp_out);

    correct_weights();

    ret = true;

    NNLOG_MIL("Backward Propogation Complete");

    return ret;

}

float NeuralNet::apply_act_func(float n)
{
    float ret = 0.0;
    if(!initialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return ret;
    }

    switch(m_act_func)
    {
        case SIGMOID:
            ret = sigmoidf(n);

        default:
            break;
    }

    return ret;
}

float NeuralNet::apply_act_func_derv(float n)
{
    float ret = 0.0;
    if(!initialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return ret;
    }

    switch(m_act_func)
    {
        case SIGMOID:
            ret = derivative_sigmoidf(n);

        default:
            break;
    }

    return ret;
}

float NeuralNet::find_delta_of_all_nodes_and_correct_biases(float* exp_out)
{

    if(!initialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return -1.0;
    }

    uint i = 0; //layer index
    uint j = 0; //current layer node index
    uint k = 0; //previous layer node index

    float* error = (float*)malloc(m_lys[num_lys - 1]->get_num_nodes() * sizeof(float));

    float total_error = calculate_error(exp_out, error);

    float temp = 0.0;

    // sigma += out_lyr->get_node_bias_idx(j);
    // sigma /= in_lyr->get_num_nodes();
    // NNLOG_DEBUG("layer [%d]: node[%d]:net=%f", in_layer_idx + 1, j, sigma);
    // sigma = apply_act_func(sigma);
    // NNLOG_DEBUG("layer [%d]: node[%d]=%f", in_layer_idx + 1, j, sigma);
    // out_lyr->set_node_value(sigma, j);

    for(i = (num_lys - 1); i > INPUT_LAYER_ID; i--)
    {
        if(m_lys[i]->get_layer_type() == OUTPUT_LYR)//output layer
        {
            for(j = 0; j < m_lys[i]->get_num_nodes(); j++)
            {
                temp = -1.0 * (exp_out[j] - m_lys[i]->get_node_value_idx(j)); //deivative of error function
                //temp *= m_lys[i - 1]->get_num_nodes();
                NNLOG_DEBUG("-(target_Node - act_node)=%f", temp);
                temp *= apply_act_func_derv(m_lys[i]->get_node_value_idx(j));
                NNLOG_DEBUG("apply_act_func_derv=%f val:%f", apply_act_func_derv(m_lys[i]->get_node_value_idx(j)), m_lys[i]->get_node_value_idx(j));
                m_lys[i]->set_node_delta(temp, j);
                NNLOG_DEBUG("layer[%d] node[%d] bias=%f delta=%f learningRate=%f", i, j, m_lys[i]->get_node_bias_idx(j), temp, learningRate);
                m_lys[i]->set_node_bias((m_lys[i]->get_node_bias_idx(j) - (learningRate * temp)), j);
            }
        }
        else //hidden layer
        {
            for(j = 0; j < m_lys[i]->get_num_nodes(); j++)
            {
                temp = 0.0;

                for(k = 0; k < m_lys[i + 1]->get_num_nodes(); k++)
                {
                    temp += m_lys[i + 1]->get_node_delta_idx(k) * m_wt_mtcs[i]->get_weight(j, k);
                    NNLOG_DEBUG("prev layer[%d] node[%d] deltas:%f wt:%f", i + 1, k,m_lys[i + 1]->get_node_delta_idx(k), m_wt_mtcs[i]->get_weight(j, k));
                }
                //temp *= m_lys[i - 1]->get_num_nodes();
                NNLOG_DEBUG("sum of prev layer nodes deltas * appropriate wts:%f node_val:%f", temp, m_lys[i]->get_node_value_idx(j));
                temp *= apply_act_func_derv(m_lys[i]->get_node_value_idx(j));
                m_lys[i]->set_node_delta(temp, j);
                NNLOG_DEBUG("layer[%d] node[%d] bias=%f delta=%f learningRate=%f apply_act_func_derv:%f", i, j, m_lys[i]->get_node_bias_idx(j), temp, learningRate, apply_act_func_derv(m_lys[i]->get_node_value_idx(j)));
                m_lys[i]->set_node_bias((m_lys[i]->get_node_bias_idx(j) - (learningRate * temp)), j);
            }
        }
        

    }

    free(error);

    return total_error;

}

void NeuralNet::correct_weights()
{
    if(!initialized)
    {
        NNLOG_ERR("exiting, not initialized!!!");
        return;
    }

    
    uint i; //in layer index, out layer index is always in layer index + 1
    uint j; //in layer node index
    uint k; //out layer node index

    float delta_wt = 0.0;

    for(i = INPUT_LAYER_ID; i < num_lys - 1; i++)
    {
        for(j = 0; j < m_lys[i]->get_num_nodes(); j++)
        {
            for(k = 0; k < m_lys[i + 1]->get_num_nodes(); k++)
            {
                delta_wt = m_lys[i + 1]->get_node_delta_idx(k) * m_lys[i]->get_node_value_idx(j);
                NNLOG_DEBUG("layer[%d] wt[%d][%d] out_node_delta=%f in_node_val=%f learningRate=%f old_wt:%f", i, j, k,  m_lys[i + 1]->get_node_delta_idx(k), m_lys[i]->get_node_value_idx(j), learningRate, m_wt_mtcs[i]->get_weight(j, k));
                delta_wt *= learningRate;

                m_wt_mtcs[i]->set_weight(j, k, (m_wt_mtcs[i]->get_weight(j, k) - delta_wt));

                NNLOG_DEBUG("layer[%d] wt[%d][%d] new_wt=%f delta_wt=%f", i, j, k,  m_wt_mtcs[i]->get_weight(j, k), delta_wt);
            }
            

        }
    }    

}

bool NeuralNet::Train(float* in, float* out)
{
    bool ret = false;

    forward_propagation(in);

    ret = isCorrectPrediction(out);

    backward_propogation(out);

    return ret;
}

void NeuralNet::populateWeightsAndBiasesWithRandomNumbers()
{
    uint i = 0;

    for(i = 0; i < num_lys; i++)
    {
        if(i != (num_lys - 1))
        {
            m_wt_mtcs[i]->populateWeightsWithRandomNumbers();
        }
        m_lys[i]->populateBiasesWithRandomNumbers();
    }
}

bool NeuralNet::isCorrectPrediction(float* out)
{
    bool ret = false;

    uint out_correct_label = 0;

    uint highest_out_label = 0;

    float highest_value = 0.0;

    uint i = 0;

    for(i = 0; i < m_lys[num_lys - 1]->get_num_nodes(); i++)
    {
        if(out[i] == 1.0)
        {
            out_correct_label = i;
        }

        if(m_lys[num_lys - 1]->get_node_value_idx(i) > highest_value)
        {
            highest_value = m_lys[num_lys - 1]->get_node_value_idx(i);
            highest_out_label = i;
        }

    }

    if(highest_out_label == out_correct_label)
    {
        ret = true;
    }

    return ret;

}

bool NeuralNet::Test(float* in, float* out)
{

    bool ret = false;

    forward_propagation(in);

    ret = isCorrectPrediction(out);

    return ret;

}

