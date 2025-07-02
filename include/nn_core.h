#ifndef NN_CORE
#define NN_CORE

#include <stdlib.h>
#include <cstdlib>
#include <unistd.h>
#include <thread>
#include <cassert>
#include "nn_math.h"

#include "baseLayer.h"

#include "baseOptimizer.h"

#include "baseLossFunction.h"

#include <vector>




#define INPUT_LAYER_ID 0 //input layer is the first layer




/*
    list of activation functions
    Note : keep ADAM in bottom to keep tests intact
*/
enum eOptimizers{
    SGD,
    RMSPROP,
    ADAM
};


//when below changed make sure to update get set save and load in nn_core
struct nnInitData{

    uint unNoLys = 0;
    std::vector<sLayer_Dimensions> layer_dimensions;
    std::vector<eAct_func> eAct_Funcs;
    std::vector<eLayer_type> e_layer_type;
    float fLearningRate = 0.5f;
    uint ID = 0;
    eOptimizers eOpt = eOptimizers::SGD;
    eLossFuncs eLossFunc = eLossFuncs::MSE;
    float optParam1 = 0.0f; //RMS_PROP : beta, ADAM : beta1
    float optParam2 = 0.0f; //RMS_PROP : epsilon, ADAM : beta2
    float optParam3 = 0.0f; //ADAM : epsilon
    std::vector<float>actParam1; //LEAKY_RELU : delta
    float lossParam1 = 0.0f; //HUBER : delta
    //ToDo: parameters for actFunc and Optimizers
    
    nnInitData(uint NumLys)
    {
        unNoLys = NumLys;
        layer_dimensions.reserve(unNoLys);
        eAct_Funcs.reserve(unNoLys);
        e_layer_type.reserve(unNoLys);
        actParam1.reserve(unNoLys);
    };

    ~nnInitData()
    {
    }

};


class NeuralNet{

    

    BaseLayer** m_ppLys; //starting address of layers

    uint m_unNumLys; //total number of layers in NN, including input and output layer

    float m_fLearningRate; //current learning rate of nn

    uint m_unTotalCorrectableNodes;

    //Optimizer
    eOptimizers m_eOpt = eOptimizers::SGD;
    BaseOptimizer* m_pOptimizer;
    float m_optParam1;
    float m_optParam2;
    float m_optParam3;

    
    //Loss Function
    eLossFuncs m_eLossFunc;
    BaseLossFunction* m_pLossFunc;
    float m_lossParam1;

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

    // Constructor to load pre-existing data
    NeuralNet(const char* fileName);

    /* 
        Destructor
    */
    ~NeuralNet();

    void Get_OutputLayer_Data(float* fVal);

    void Get_Init_Data(nnInitData *ret);

    void Init_Batch_Training(uint batchSz);

    /*
        do_forward_pass() : perform 1 iteration of forward pass

        @input_arr : array containing input node values
    */
    bool do_forward_pass(float* pfInputArr);


    /*
        populate_weights() : populates weight matrix of particular index with specific values. Useful during initialization

        @idx : index of matrix
        @values : list of values to populate
    */
    void populate_weights(uint unIdx, float* pfValues);

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

    void SetBias(uint LayerID, uint NodeID, float val);

    float GetLearningRate();

    float GetDelta(uint LayerID, uint NodeID);

    float GetNodeVal(uint LayerID, uint NodeID);

    void SetWeight(uint MtxId, uint inIdx, uint outIdx, float val); 

    void SaveNN(const char* fileName);   

    BaseLayer* GetLayer(uint idx);

    const float* GetMatrix(uint idx);

    BaseLossFunction* GetLossFunc();

    eLayer_type get_layer_type(uint idx);




    private:
     
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

    void SetupLayersAndWeightMatrices(std::vector<eLayer_type>& layer_types, std::vector<sLayer_Dimensions>& dims, std::vector<eAct_func>& actFuncs, std::vector<float>& actParam1, float lossParam);

    void MergeBiasAndWeights(uint i);

    void setup_dense_layer(uint layer_idx, uint out_sz, eAct_func act_func, float actParam);

    void setup_conv_layer(uint layer_idx, uint input_rows, uint input_columns, uint kernel_rows, uint kernel_columns, eAct_func act_func, float actParam);

    
};

#endif