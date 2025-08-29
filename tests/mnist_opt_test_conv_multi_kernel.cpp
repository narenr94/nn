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

    uint sz[4] = {784,2028,32,10};

    eAct_func actFuncs[4] = {eAct_func::TANH, eAct_func::TANH, eAct_func::TANH, eAct_func::TANH};

    std::vector<float> norm_values(VAL_SIZE);
    std::vector<float> out(10);
    uint u_sz = 4;

    nnInitData * initData = new nnInitData(u_sz);

    initData->unNoLys = 4;

    initData->e_layer_type[0] = eLayer_type::INPUT;
    initData->layer_dimensions[0].unInputColumns = 0;
    initData->layer_dimensions[0].unInputRows = 0;
    initData->layer_dimensions[0].unNoTransformParameterMtx = 0;
    initData->layer_dimensions[0].unOutputColumns = 784;
    initData->layer_dimensions[0].unOutputRows = 1;
    initData->layer_dimensions[0].unTransformParametersColumns = 0;
    initData->layer_dimensions[0].unTransformParametersRows = 0;

    
    initData->e_layer_type[1] = eLayer_type::CONV;
    initData->layer_dimensions[1].unInputColumns = 28;
    initData->layer_dimensions[1].unInputRows = 28;
    initData->layer_dimensions[1].unNoTransformParameterMtx = 3;
    initData->layer_dimensions[1].unOutputColumns = 26;
    initData->layer_dimensions[1].unOutputRows = 78;
    initData->layer_dimensions[1].unTransformParametersColumns = 3;
    initData->layer_dimensions[1].unTransformParametersRows = 3;


    initData->e_layer_type[2] = eLayer_type::DENSE;
    initData->layer_dimensions[2].unInputColumns = 2028;
    initData->layer_dimensions[2].unInputRows = 1;
    initData->layer_dimensions[2].unNoTransformParameterMtx = 1;
    initData->layer_dimensions[2].unOutputColumns = 32;
    initData->layer_dimensions[2].unOutputRows = 1;
    initData->layer_dimensions[2].unTransformParametersColumns = 32;
    initData->layer_dimensions[2].unTransformParametersRows = 2028;

    initData->e_layer_type[3] = eLayer_type::DENSE;
    initData->layer_dimensions[3].unInputColumns = 32;
    initData->layer_dimensions[3].unInputRows = 1;
    initData->layer_dimensions[3].unNoTransformParameterMtx = 1;
    initData->layer_dimensions[3].unOutputColumns = 10;
    initData->layer_dimensions[3].unOutputRows = 1;
    initData->layer_dimensions[3].unTransformParametersColumns = 10;
    initData->layer_dimensions[3].unTransformParametersRows = 32;

/*
Observation
Adam doesnt seem to be working with default values for beta1, beta2 and epsilon
from trial and error found that beta1=0.9f, beta2 = 0.9f and epsilon = 0.1f works
RMF Prop sometimes works with default others it works with below
beta = 0.999f epsilon = 0.00000001f
idhu oru manda kolaru bro
*/
    //SGD
    // initData->eOpt = eOptimizers::SGD;
    // initData->fLearningRate = 0.01f;
    //ADAM
    // initData->eOpt = eOptimizers::ADAM;
    // initData->optParam1 = 0.9f;
    // initData->optParam2 = 0.9f;
    // initData->optParam3 = 0.1f;
    // initData->fLearningRate = 0.01f;
    //RMSPROP
    initData->eOpt = eOptimizers::RMSPROP;
    initData->optParam[0] = 0.999f;
    initData->optParam[1] = 0.00000001f;
    initData->fLearningRate = 0.1f;


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