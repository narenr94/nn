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
    
    initData->fLearningRate = 0.01f;

    NeuralNet *nn = new NeuralNet(*initData);

    Init_NN_for_MNIST(nn);
    
    for(j = 0; j < EPOCH_MAX; j++)
    {
        printf("Epoch[%d] Started!!!\n", j + 1);
        Train_NN_for_MNIST(nn);
        Test_NN_for_MNIST(nn);
    }

    printf("start save\n");

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> time_taken;

    start = std::chrono::high_resolution_clock::now();

    std::string sav_loc = "MNIST_EPOCH5.sav";

    nn->Save_NN(sav_loc);

    end = std::chrono::high_resolution_clock::now();

    time_taken = end - start;

    printf("Time taken to save file:%fSeconds\n", time_taken.count());

    printf("done save\n");

    
    
    printf("start load and init\n");
    start = std::chrono::high_resolution_clock::now();

    NeuralNet *nn2 = new NeuralNet(sav_loc);

    end = std::chrono::high_resolution_clock::now();

    time_taken = end - start;

    printf("Time taken to load and init nn from file:%fSeconds\n", time_taken.count());

    printf("done load and init\n");

    printf("\nTest Loaded NN\n");

    Test_NN_for_MNIST(nn2);

    delete nn2;

    delete initData;

    return 0;

}