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
    eLossFuncs lossVal;
    
    
    for(uint k = 0; k < 4; k++)
    {

        uint u_sz = 4;

        nnInitData * initData = new nnInitData(u_sz);

        switch(k)
        {
            case 0:
                lossVal = eLossFuncs::MSE;
                break;
            case 1:
                lossVal = eLossFuncs::MAE;
                break;
            case 2:
                lossVal = eLossFuncs::HUBER;
                break;
            default:
                lossVal = eLossFuncs::MSE;
                break;
        }
        
        float correct_count = 0;

        float accuracy = 0.0;   

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
        // initData->eAct_Func = eAct_func::TANH;
        initData->fLearningRate = 0.01f;
        initData->eLossFunc = lossVal;
        initData->eOpt = eOptimizers::SGD;

        switch(lossVal)
        {
            case eLossFuncs::MSE:
                printf("\n Loss Func MSE \n");
                break;
            case eLossFuncs::MAE:
                printf("\n Loss Func MAE \n");
                initData->fLearningRate = 0.001f;
                break;
            case eLossFuncs::HUBER:
                printf("\n Loss Func HUBER \n");
                break;

        }

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
    }

    return 0;

}