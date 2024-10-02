#ifndef NN_CORE
#define NN_CORE

#include <stdlib.h>
#include <cstdlib>
#include "nn_math.h"
#include "nn_l2l_weight_matrix.h"


#define INPUT_LAYER_ID 0 //input layer is the first layer

#define MAX_DUMP_FILE_NUM_STR_SIZE 3 //size of dump file num str including '\0' terminator

#define MAX_DUMP_FILE_NAME_STR_SIZE 20 //size of dump file name str including '\0' terminator


/*
    list of activation functions
*/
enum eAct_func{
    SIGMOID
};

class NeuralNet{

    eAct_func m_eActFunc; //activation funcation to be used

    nn_l2l_weight_matrix** m_ppWtMtcs; //starting address of weight matrixes

    nn_layer** m_ppLys; //starting address of layers

    uint m_unNumLys; //total number of layers in NN, including input and output layer

    bool m_bInitialized; //is neural net initialized?

    float m_fLearningRate; //current learning rate of nn


    
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
    bool init(uint unNoLys, uint* unSzLys, eAct_func eActFunc, elog_level eLogLevel, bool bConsolePrint, float fLearningRate);

    /*
        do_forward_pass() : perform 1 iteration of forward pass

        @input_arr : array containing input node values
    */
    bool do_forward_pass(float* pfInputArr);

    /*
        dump_nn() : dumps current nn contents into dmp file
    */
    void dump_nn();

    /*
        populate_weights() : populates weight matrix of particular index with specific values. Useful during initialization

        @idx : index of matrix
        @values : list of values to populate
    */
    bool populate_weights(uint unIdx, float* pfValues);

    /*
        populate_nodes() : populates nodes of particular layer with specific values and biases. Useful during initialization

        @lyr_idx : index of layer
        @biases : list of biases to populate
    */
    bool populate_nodes_bias(uint unLyrIdx, float* pfBias);

    /*
        calculate_error() : calculates error from expected and actual output. returns total error

        @exp_out : expected output array
        @error : array where error ought to be stored
    */
    float calculate_error(float* pfExpOut, float* pfError);

    /*
        backward_propogation() : backward propogate once

        @exp_out : expected output array
    */
    bool do_backward_pass(float* pfExpOut);
    
    /*
        Train() : train neural net

        @in : input array
        @out : expected output array
    */
    bool Train(float* pfIn, float* pfOut);

    /*
        populateWeightsAndBiasesWithRandomNumbers() : fill weights and biases with random numbers
    */
    void populateWeightsAndBiasesWithRandomNumbers();    

    /*
        Test() : tes neural net

        @in : input array
        @out : expected output array
    */
    bool Test(float* pfIn, float* pfOut);
    

    private:
    /*
        forwardpass_to_next_layer() : perform forward pass from (in_layer_idx)th layer to (in_layer_idx + 1)th layer

        @in_layer_idx : input (relative) layer index
    */
    bool do_forwardpass_to_next_layer(uint unInLayerIdx);

    /*
        apply_act_func() : applies act function as per config

        @n : value to which act func is to be applied
    */
    float apply_act_func(float fVal);

    /*
        apply_act_func_derv() : applies act function derivative as per config

        @n : value to which act func derivative is to be applied
    */
    float apply_act_func_derv(float fVal);

    /*
        find_delta_of_all_nodes() : calculates and updates delta of all nodes in nn, also corrects biases

        @exp_out : array containing expected output for nn

        @return : total error
    */
    float find_delta_of_all_nodes_and_correct_biases(float* pfExpOut);

    /*
        correct_weights() : corrects all weights
    */
    void correct_weights();

    /*
        isCorrectPrediction() : compares neural net output with expected output and return true is correct.

        @out : expected output array
    */
    bool isCorrectPrediction(float* pfOut);   

};

#endif