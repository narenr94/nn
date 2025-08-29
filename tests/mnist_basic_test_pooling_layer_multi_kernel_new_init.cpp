#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "nn_core.h"
#include "nn_utils.h"
#include <chrono>

#include "nn_common_utils.h"

#define EPOCH_MAX 10 //number epochs of training and testing 

int main()
{
    uint i = 0;

    uint j = 0;

    uint sz[4] = {784,2028,32,10}; //28x28
    //26x26
    eAct_func actFuncs[4] = {eAct_func::TANH, eAct_func::TANH, eAct_func::TANH, eAct_func::TANH};

    std::vector<float> norm_values(VAL_SIZE);

    std::vector<float> out(10);

    std::vector<float> optParam;
    optParam.push_back(0.0f);
    optParam.push_back(0.0f);
    optParam.push_back(0.0f);

    sMtx_Dim in_dim;
    in_dim.rows = 28;
    in_dim.columns = 28;

    nnInitData * initData = new nnInitData(in_dim, eOptimizers::RMSPROP, optParam, eLossFuncs::CCE, 0.0f, 0.001f);
    in_dim = initData->add_conv_layer(in_dim, eConvKernelSize::Sz3x3, 16, eAct_func::RELU, 0.0f);
    initData->add_pooling_layer(in_dim, ePooling_type::MAX, ePoolingKernelSize::KrSz2x2);
    initData->add_dense_layer(10, eAct_func::SOFTMAX, 0.0f);

    NeuralNet *nn = new NeuralNet(*initData);

    Init_NN_for_MNIST(nn);

    for(j = 0; j < EPOCH_MAX; j++)
    {
        printf("Epoch[%d] Started!!!\n", j + 1);
        
        Train_NN_for_MNIST(nn);

        Test_NN_for_MNIST(nn);
    }
    
    delete nn;

    delete initData;

    return 0;

}