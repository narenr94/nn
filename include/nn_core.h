#ifndef NN_CORE
#define NN_CORE

#include <stdlib.h>
#include <cstdlib>
#include "nn_math.h"
#include "nn_logger.h"


#define INPUT_LAYER_ID 0 //input layer is the first layer

#define MAX_DUMP_FILE_NUM_STR_SIZE 3 //size of dump file num str including '\0' terminator

#define MAX_DUMP_FILE_NAME_STR_SIZE 20 //size of dump file name str including '\0' terminator

#define RAND_MIN_WEIGHT_BIAS 1

#define RAND_MAX_WEIGHT_BIAS 9

using namespace std;

/*
    list of activation functions
*/
enum eAct_func{
    SIGMOID
};

/*
    list of layer types
*/
enum eLyr_type{
    INPUT_LYR,
    HIDDEN_LYR,
    OUTPUT_LYR
};

class nn_node{

    float value; //value of node
    float bias; //value of bias

    float delta; //delta value of node , used for back propogation

    
    

    public:

    /*
        nn_node : constructor, sets value and bias to 0.0
    */
    nn_node();

    /*
        set_value : Sets value of node
        @n : value to be set
    */
    void set_value(float n);
    /*
        set_bias : Sets bias of node
        @n : value of bias to be set
    */
    void set_bias(float n);
    /*
        set_delta : Sets delta of node
        @n : value of delta to be set
    */
    void set_delta(float n);
    /*
        get_delta() : returns current delta of node
    */
    float get_delta();
    /*
        get_value() : returns current value of node
    */
    float get_value();
    /*
        get_bias() : returns current bias value of node
    */
    float get_bias();

};

class nn_layer{

    uint num_nodes; //total number of nodes
    nn_node** nodes; //starting address from nodes can be accessed
    bool initialized; //is layer initialized?
    eLyr_type lyrType; //layer type


    
    public:

    /*
        nn_layer() : constructor for layer
        @n_nodes : number of nodes in layer
    */
    nn_layer(uint n_nodes);

    /*
        ~nn_layer() : destruct and frees layer resources
    */
    ~nn_layer();

    /*
        get_num_nodes() : returns total number of nodes in layer
    */
    uint get_num_nodes();

    /*
        set_node_value() : sets value of node in particular index

        @value : value to be set in node
        @index : the index represnting the node where value is to be set
    */
    bool set_node_value(float value, uint index);

    /*
        set_all_node_values() : sets value of all nodes in layer

        @value : array containing values to be added
        
    */
    bool set_all_node_values(float* value);

    /*
        set_all_node_biases() : sets value of all nodes in layer

        @bias : array containing bias to be added
        
    */
    bool set_all_node_biases(float* bias);

    /*
        set_node_bias() : sets bias of node in particular index

        @bias : bias to be set in node
        @index : the index represnting the node where bias is to be set
    */
    bool set_node_bias(float bias, uint index);

    /*
        set_node_delta() : sets delta of node in particular index

        @delta : delta to be set in node
        @index : the index represnting the node where bias is to be set
    */
    bool set_node_delta(float bias, uint index);

    /*
        is_initialized() : returns if layer is initialized or not
    */
    bool is_initialized();

    /*
        get_node_value_idx() : returns value of node in specified index

        @idx : the index represnting the node from where value is to be got
    */
    float get_node_value_idx(uint idx);
    /*
        get_node_bias_idx() : returns bias of node in specified index

        @idx : the index represnting the node from where bias is to be got
    */
    float get_node_bias_idx(uint idx);

    /*
        get_node_delta_idx() : returns delta of node in specified index

        @idx : the index represnting the node from where delta is to be got
    */
    float get_node_delta_idx(uint idx);

    /*
        set_layer_type() : sets layer type

        @lt : layer type to be set
    */
    void set_layer_type(eLyr_type lt);

    /*
        get_layer_type() : gets layer's type
    */
    eLyr_type get_layer_type();
    
    /*
        populateBiasesWithRandomNumbers() : assign random numbers to biases
    */
    void populateBiasesWithRandomNumbers();


};

class nn_l2l_weight_matrix{
    
    nn_layer* input_layer; //input layer for matrix
    nn_layer* output_layer; //output layer for matrix

    uint size; //size of matrix

    float* weight_matrix; //starting address from where weights of matrix are stored

    bool initialized; //is weight matrix initialized?

    
    public:

    /*
        nn_l2l_weight_matrix() : constructor of layer to layer weight matrix
        @in : pointer to input/left layer
        @ot : pointer to output/right layer
    */
    nn_l2l_weight_matrix(nn_layer* in, nn_layer* ot);

    /*
        ~nn_l2l_weight_matrix() : destructs and frees resources of layer to layer weight matrix
    */
    ~nn_l2l_weight_matrix();

    /*
        set_weight() : sets weight of connection between particulat input and output node

        @in_idx : input node index
        @out_idx : output node index
        @wt : value of weight
    */
    bool set_weight(uint in_idx, uint out_idx, float wt);

    /*
        set_all_weight() : sets all weight in matrix

        @wt : array of values of weights
    */
    bool set_all_weight(float* wt);

    /*
        get_weight() : gets weight of connection between particulat input and output node

        @in_idx : input node index
        @out_idx : output node index
    */
    float get_weight(uint in_idx, uint out_idx);

    /*
        get_size() : gets size of matrix
    */
    uint get_size();

    /*
        populateWeightsWithRandomNumbers() : assign random numbers to weights
    */
    void populateWeightsWithRandomNumbers();


};

class NeuralNet{

    eAct_func m_act_func; //activation funcation to be used

    nn_l2l_weight_matrix** m_wt_mtcs; //starting address of weight matrixes

    nn_layer** m_lys; //starting address of layers

    uint num_lys; //total number of layers in NN, including input and output layer

    bool initialized; //is neural net initialized?

    float learningRate; //current learning rate of nn


    
    public:

    /*
        Constructor
    */
    NeuralNet();

    /* 
        Destructor
    */
    ~NeuralNet();

    /*
        init() : sets neural network

        @no_lys : total number of layers, including input and output layer
        @sz_lys : array containing size of each layer, size of this array must be equal to total number of layers
    */
    bool init(uint no_lys, uint* sz_lys, eAct_func act_func, elog_level loglevel, bool consolePrint, float lRate);

    /*
        forward_propagation() : perform 1 iteration of forward pass

        @input_arr : array containing input node values
    */
    bool forward_propagation(float* input_arr);

    /*
        dump_nn() : dumps current nn contents into dmp file
    */
    void dump_nn();

    /*
        populate_weights() : populates weight matrix of particular index with specific values. Useful during initialization

        @idx : index of matrix
        @values : list of values to populate
    */
    bool populate_weights(uint idx, float * values);

    /*
        populate_nodes() : populates nodes of particular layer with specific values and biases. Useful during initialization

        @lyr_idx : index of layer
        @biases : list of biases to populate
    */
    bool populate_nodes_bias(uint lyr_idx, float* bias);

    /*
        calculate_error() : calculates error from expected and actual output. returns total error

        @exp_out : expected output array
        @error : array where error ought to be stored
    */
    float calculate_error(float* exp_out, float* error);

    /*
        backward_propogation() : backward propogate once

        @exp_out : expected output array
    */
    bool backward_propogation(float* exp_out);
    
    /*
        Train() : train neural net

        @in : input array
        @out : expected output array
    */
    bool Train(float* in, float* out);

    /*
        populateWeightsAndBiasesWithRandomNumbers() : fill weights and biases with random numbers
    */
    void populateWeightsAndBiasesWithRandomNumbers();    

    /*
        Test() : tes neural net

        @in : input array
        @out : expected output array
    */
    bool Test(float* in, float* out);
    

    private:
    /*
        forwardpass_to_next_layer() : perform forward pass from (in_layer_idx)th layer to (in_layer_idx + 1)th layer

        @in_layer_idx : input (relative) layer index
    */
    bool forwardpass_to_next_layer(uint in_layer_idx);

    /*
        apply_act_func() : applies act function as per config

        @n : value to which act func is to be applied
    */
    float apply_act_func(float n);

    /*
        apply_act_func_derv() : applies act function derivative as per config

        @n : value to which act func derivative is to be applied
    */
    float apply_act_func_derv(float n);

    /*
        find_delta_of_all_nodes() : calculates and updates delta of all nodes in nn, also corrects biases

        @exp_out : array containing expected output for nn

        @return : total error
    */
    float find_delta_of_all_nodes_and_correct_biases(float* exp_out);

    /*
        correct_weights() : corrects all weights
    */
    void correct_weights();

    /*
        isCorrectPrediction() : compares neural net output with expected output and return true is correct.

        @out : expected output array
    */
    bool isCorrectPrediction(float* out);   

};

#endif