#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "nn_core.h"
#include "nn_utils.h"
#include <chrono>

#include "nn_common_utils.h"

#define EPOCH_MAX 3 //number epochs of training and testing 

int main()
{
    uint i = 0;

    uint j = 0;

    uint sz[4] = {784,32,32,10};
    eAct_func actFuncs[4] = {eAct_func::TANH, eAct_func::TANH, eAct_func::TANH, eAct_func::TANH};

    std::vector<float> norm_values(VAL_SIZE);
    std::vector<float> out(10);
    uint u_sz = 4;

    nnInitData * initData = new nnInitData(u_sz); 

    initData->unNoLys = 4;

    set_layer_info(sz, initData->unNoLys, initData);

    for(uint l = 0; l < initData->unNoLys; l++)
    {
        if(l != 0)
        {
            initData->e_layer_type[l] = eLayer_type::DENSE;
        }
        else
        {
            initData->e_layer_type[l] = eLayer_type::INPUT;
        }
        initData->eAct_Funcs[l] = actFuncs[l];
    }

/*
Observation
Adam doesnt seem to be working with default values for beta1, beta2 and epsilon
from trial and error found that beta1=0.9f, beta2 = 0.9f and epsilon = 0.1f works
RMF Prop sometimes works with default others it works with below
beta = 0.999f epsilon = 0.00000001f
idhu oru manda kolaru bro
*/
    initData->eOpt = eOptimizers::RMSPROP;
    //SGD
    // initData->eOpt = eOptimizers::SGD;
    // initData->fLearningRate = 0.01f;
    //ADAM
    initData->eOpt = eOptimizers::ADAM;
    initData->optParam[0] = 0.9f;
    initData->optParam[1] = 0.99f;
    initData->optParam[2] = 0.01f;
    initData->fLearningRate = 0.01f;
    //RMSPROP
    // initData->eOpt = eOptimizers::RMSPROP;
    // initData->optParam1 = 0.999f;
    // initData->optParam2 = 0.00000001f;
    // initData->fLearningRate = 0.1f;


    initData->eLossFunc = eLossFuncs::HUBER;

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