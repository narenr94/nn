#ifndef NN_CORE
#define NN_CORE

#include <stdlib.h>
#include <cstdlib>
#include <unistd.h>
#include <thread>
#include <cassert>
#include <vector>

#include "nn_math.h"
#include "baseLayer.h"
#include "baseOptimizer.h"
#include "baseLossFunction.h"
#include "nn_defines.h"


#define INPUT_LAYER_ID 0 //input layer is the first layer


class NeuralNet{

    

    BaseLayer** m_ppLys = nullptr; //starting address of layers

    uint m_unNumLys = 0; //total number of layers in NN, including input and output layer

    float m_fLearningRate = 0.01f; //current learning rate of nn

    uint m_unTotalCorrectableNodes = 0;

    //Optimizer
    eOptimizers m_eOpt = eOptimizers::SGD;
    BaseOptimizer* m_pOptimizer = nullptr;
    float m_optParam1 = 0.0f;
    float m_optParam2 = 0.0f;
    float m_optParam3 = 0.0f;

    
    //Loss Function
    eLossFuncs m_eLossFunc = eLossFuncs::HUBER;
    BaseLossFunction* m_pLossFunc = nullptr;
    float m_lossParam1 = 0.0f;

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
    NeuralNet(nnInitData& initData);

    // Copy constructor
    NeuralNet(NeuralNet* other);

    // Constructor to load pre-existing data
    NeuralNet(std::string& fileName);

    /* 
        Destructor
    */
    ~NeuralNet();

    void Get_OutputLayer_Data(float* fVal);

    nnInitData Get_Init_Data();

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

    void SetWeight(uint MtxId, uint Idx, float val);

    void Save_NN(std::string fileName);   

    BaseLayer* GetLayer(uint idx);

    const float* GetMatrix(uint idx);

    BaseLossFunction* GetLossFunc();

    eLayer_type get_layer_type(uint idx);


    private:
     
    bool isCorrectPrediction(float* pfOut);   

    void Batch_Training(uint i);

    void Populate_Batch_Processing_Args(float** in, float** out, uint numIn);

    void Set_Init_Data(nnInitData& other_initData);

    void SetupLayersAndWeightMatrices(std::vector<eLayer_type>& layer_types, std::vector<sLayer_Dimensions>& dims, std::vector<eAct_func>& actFuncs, std::vector<float>& actParam1, float lossParam, std::vector<ePooling_type>& ePoolingType);

    void MergeBiasAndWeights(uint i);

    void Set_Load_Data(std::vector<std::string> blocks);

    void Set_General_Data(sNN_General_Data gen_data);

    void Set_Layer_Data(sNN_General_Data gen_data, std::vector<std::string> blocks);

    void Set_Layer_Order(float m_lossParam);
    
};

#endif