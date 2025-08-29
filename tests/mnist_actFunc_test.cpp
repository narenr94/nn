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

    uint sz[5] = {784,32,32,10};

    std::vector<float> norm_values(VAL_SIZE);

    std::vector<float> out(10);

    for(uint k = 0; k <= (uint)(eAct_func::TANH); k++)
    {
        eAct_func actVal = (eAct_func)k;
        float correct_count = 0;

        float accuracy = 0.0;   

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
            initData->eAct_Funcs[l] = eAct_func::TANH;
            if(l == (initData->unNoLys - 1))
            {
                initData->eAct_Funcs[l] = actVal;
            }
        }
        
        initData->fLearningRate = 0.01f;
        initData->eOpt = eOptimizers::SGD;

        initData->eLossFunc = eLossFuncs::MSE; 

        switch(actVal)
        {
            case eAct_func::SIGMOID:
                initData->fLearningRate = 0.5f;
                printf("\n Act Func SIGMOID \n");
                break;
            case eAct_func::RELU:
                printf("\n Act Func RELU \n");
                break;
            case eAct_func::LEAKY_RELU:
                printf("\n Act Func LEAKY_RELU \n");
                break;
            case eAct_func::TANH:
                printf("\n Act Func TANH \n");
                break;
            case eAct_func::SOFTMAX:
                printf("\n Act Func SOFTMAX \n");
                initData->eLossFunc = eLossFuncs::CCE;
                break;
            default:
                assert(0); //unknown activation function
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