#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "nn_core.h"
#include "nn_utils.h"
#include <chrono>

#include "nn_common_utils.h"

#define EPOCH_MAX 1 //number epochs of training and testing 


int main()
{
    uint i = 0;

    uint j = 0;

    uint sz[4] = {784,32,32,10};


    eAct_func actFuncs[4] = {eAct_func::TANH, eAct_func::TANH, eAct_func::TANH, eAct_func::TANH};

    std::vector<float> norm_values(VAL_SIZE);
    std::vector<float> out(10);

    uint u_sz = 4;

    std::vector<float> optParam;
    optParam.push_back(0.0f);
    optParam.push_back(0.0f);
    optParam.push_back(0.0f);

    sMtx_Dim in_dim;
    in_dim.rows = 28;
    in_dim.columns = 28;

    nnInitData * initData = new nnInitData(in_dim, eOptimizers::SGD, optParam, eLossFuncs::HUBER, 0.0f, 0.01f);
    initData->add_conv_layer(in_dim, eConvKernelSize::Sz3x3, 3, eAct_func::TANH, 0.0f);
    initData->add_dense_layer(64, eAct_func::TANH, 0.0f);
    initData->add_dense_layer(10, eAct_func::TANH, 0.0f);

    NeuralNet *nn = new NeuralNet(*initData);

    Init_NN_for_MNIST(nn);
    
    for(j = 0; j < EPOCH_MAX; j++)
    {
        printf("Epoch[%d] Started!!!\n", j + 1);

        Train_NN_for_MNIST(nn);
        Test_NN_for_MNIST(nn);
    }

    printf("\nCOPYING\n");
    NeuralNet *nn2 = new NeuralNet(nn);

    printf("\nTest new NN\n");
    Test_NN_for_MNIST(nn2);

    delete nn;

    delete nn2;

    delete initData;

    return 0;

}