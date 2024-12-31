#ifndef NN_CORE
#define NN_CORE

#include <stdlib.h>
#include <cstdlib>
#include <unistd.h>
#include <thread>
#include "nn_math.h"
#include "nn_l2l_weight_matrix.h"

#include "activationFunction.h"

#include "optimizer.h"


#define INPUT_LAYER_ID 0 //input layer is the first layer

#define MAX_DUMP_FILE_NUM_STR_SIZE 3 //size of dump file num str including '\0' terminator

#define MAX_DUMP_FILE_NAME_STR_SIZE 20 //size of dump file name str including '\0' terminator

#define MAX_DUMP_FILE_SIZE 10000000 //10Mbytes


/*
    list of activation functions
*/
enum eAct_func{
    RELU,
    LEAKY_RELU,
    TANH,
    SIGMOID
};

/*
    list of activation functions
*/
enum eOptimizers{
    SGD,
    RMSPROP,
    ADAM
};

struct nnInitData{

    uint unNoLys = 0;
    uint* unSzLys = nullptr;
    eAct_func eAct_Func = eAct_func::SIGMOID;
    float fLearningRate = 0.5f;
    uint ID = 0;
    eOptimizers eOpt = eOptimizers::SGD;
    float optParam3 = 0.0f;

    //ToDo: parameters for actFunc and Optimizers
    
    nnInitData(uint m_unNumLys)
    {
        unSzLys = new uint [m_unNumLys];
    };

    ~nnInitData()
    {
        delete [] unSzLys;
    }

};

class NeuralNet{

    eAct_func m_eActFunc; //activation funcation to be used

    nn_l2l_weight_matrix** m_ppWtMtcs; //starting address of weight matrixes

    nn_layer** m_ppLys; //starting address of layers

    uint m_unNumLys; //total number of layers in NN, including input and output layer

    bool m_bInitialized = false; //is neural net initialized?

    float m_fLearningRate; //current learning rate of nn

    uint m_unTotalCorrectableNodes;

    eOptimizers m_eOpt = eOptimizers::SGD;

    Optimizer* m_pOptimizer;

    //Activation Function
    ActivationFunction* m_pActFunc;

    //Batch Training Specific 
    bool initBatchTrain = false;

    NeuralNet ** m_batch_nns = nullptr;

    uint m_batchSz = 0;

    //debug
    // uint nn_id = 0;


    


    
    public:

    /*
        Constructor
    */
    NeuralNet(nnInitData* initData);

    // Copy constructor
    NeuralNet(NeuralNet* other);

    /* 
        Destructor
    */
    ~NeuralNet();

    void Get_Init_Data(nnInitData *ret);

    void Init_Batch_Training(uint batchSz);

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
        Train() : train neural net

        @in : input array
        @out : expected output array
    */
    uint Train_batch(float** in, float** out, uint numIn);

    /*
        populateWeightsAndBiasesWithRandomNumbers() : fill weights and biases with random numbers
    */
    void populateWeightsAndBiasesWithRandomNumbers();    

    void populateWeightsAndBiasesWithExistingNN(NeuralNet *other);

    /*
        Test() : tes neural net

        @in : input array
        @out : expected output array
    */
    bool Test(float* pfIn, float* pfOut);

    uint GetNumLys();

    uint GetSzLayer(uint LayerID);

    float GetBias(uint LayerID, uint NodeID);

    uint GetSzMtx(uint MtxID);

    float GetWeight(uint MtxID, uint Idx);

    float GetWeight(uint MtxID, uint inIdx, uint outIdx);

    void SetBias(uint LayerID, uint NodeID, float val);

    float GetLearningRate();

    float GetDelta(uint LayerID, uint NodeID);

    float GetNodeVal(uint LayerID, uint NodeID);

    void SetWeight(uint MtxId, uint inIdx, uint outIdx, float val); 


    

    private:
    
    /*
        forwardpass_to_next_layer() : perform forward pass from (in_layer_idx)th layer to (in_layer_idx + 1)th layer

        @in_layer_idx : input (relative) layer index
    */
    bool do_forwardpass_to_next_layer(uint unInLayerIdx);

    /*
        find_delta_of_all_nodes() : calculates and updates delta of all nodes in nn, also corrects biases

        @exp_out : array containing expected output for nn

        @return : total error
    */
    float find_delta_of_all_nodes(float* pfExpOut);

    
    /*
        isCorrectPrediction() : compares neural net output with expected output and return true is correct.

        @out : expected output array
    */
    bool isCorrectPrediction(float* pfOut);   

    void Batch_Training(uint i);
    void Populate_Batch_Processing_Args(float** in, float** out, uint numIn);

    // void Batch_Training();

    // void apply_delats_to_weights_and_biases_batch_training(float* deltas);

    void Set_Init_Data(nnInitData* other_initData);

    void SetupLayersAndWeightMatrices(uint *sz);

    void MergeBiasAndWeights(uint i);

};

#endif