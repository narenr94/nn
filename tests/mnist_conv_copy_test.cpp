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
    char* line_buff = (char*)malloc(BUFF_SIZE); 

    FILE* fdr = NULL;

    uint label;

    uint i = 0;

    uint j = 0;

    uint sz[4] = {784,32,32,10};


    eAct_func actFuncs[4] = {eAct_func::TANH, eAct_func::TANH, eAct_func::TANH, eAct_func::TANH};

    std::vector<float> norm_values(VAL_SIZE);
    std::vector<float> out(10);

    std::string linestr = "Line";

    nn_progress_bar *pb = new nn_progress_bar(linestr.c_str(),TRAIN_MAX);

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> time_taken;

    float correct_count = 0;

    float accuracy = 0.0;   

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
    /*
    struct nnInitData{

    uint unNoLys = 0;
    uint* unSzLys = nullptr;
    eAct_func eAct_Func = eAct_func::SIGMOID;
    elog_level eLogLevel = elog_level::eLOGLEVEL_WARN;
    bool bConsolePrint = false;
    float fLearningRate = 0.5f;

    };
    */

    NeuralNet *nn = new NeuralNet(*initData);

    nn->populateWeightsAndBiasesWithRandomNumbers();

    std::thread pbThread(&nn_progress_bar::print_progress_bar_periodic, pb, 0, 1000);
    
    for(j = 0; j < EPOCH_MAX; j++)
    {
        start = std::chrono::high_resolution_clock::now();

        printf("Epoch[%d] Started!!!\n", j + 1);
        pb->setMax(TRAIN_MAX);
        fdr = fopen("MNIST/mnist_train.csv","r");
        //getNextLine(fdr, line_buff);

        for(i = 0; i < TRAIN_MAX; i++)
        {
            
            if(!fdr)
            {
                printf("fdr open fail!!!\n");
                return 0;
            }

            getNextLine(fdr, line_buff);

            label = parseLabelAndNormalizedValues(line_buff, norm_values, NORM_FACTOR);

            setOutArray(label, out);            

            if(nn->Train(norm_values, out))
            {
                correct_count += 1.0;
            }

            pb->update_progress_bar(i + 1);            

        }

        fclose(fdr);

        pb->reset();

        accuracy = correct_count / ((float)TRAIN_MAX);

        printf("\nTrain Accuracy:%f\n", accuracy);
        end = std::chrono::high_resolution_clock::now();

        time_taken = end - start;

        printf("Time taken for Train Epoch[%d]:%fSeconds\n", j + 1, time_taken.count());
        
        correct_count = 0.0;

        pb->setMax(TEST_MAX);

        start = std::chrono::high_resolution_clock::now();

        fdr = fopen("MNIST/mnist_test.csv","r");

        //getNextLine(fdr, line_buff);

        for(i = 0; i < TEST_MAX; i++)
        {
            
            if(!fdr)
            {
                printf("fdr open fail!!!\n");
                return 0;
            }
            getNextLine(fdr, line_buff);

            label = parseLabelAndNormalizedValues(line_buff, norm_values, NORM_FACTOR);

            setOutArray(label, out);            

            if(nn->Test(norm_values, out))
            {
                correct_count += 1.0;
            }

            pb->update_progress_bar(i + 1);
        }



        fclose(fdr);

        pb->reset();

        accuracy = correct_count / ((float)TEST_MAX);

        printf("\nTest Accuracy:%f\n", accuracy);
        end = std::chrono::high_resolution_clock::now();

        time_taken = end - start;

        printf("Time taken for test Epoch[%d]:%fSeconds\n", j + 1, time_taken.count());
        
        correct_count = 0.0;

    }

    NeuralNet *nn2 = new NeuralNet(nn);

    start = std::chrono::high_resolution_clock::now();

    fdr = fopen("MNIST/mnist_test.csv","r");

    //getNextLine(fdr, line_buff);

    for(i = 0; i < TEST_MAX; i++)
    {
        
        if(!fdr)
        {
            printf("fdr open fail!!!\n");
            return 0;
        }
        getNextLine(fdr, line_buff);

        label = parseLabelAndNormalizedValues(line_buff, norm_values, NORM_FACTOR);

        setOutArray(label, out);            

        if(nn2->Test(norm_values, out))
        {
            correct_count += 1.0;
        }

        pb->update_progress_bar(i + 1);
    }



    fclose(fdr);

    pb->reset();

    accuracy = correct_count / ((float)TEST_MAX);

    printf("\nTest copy Accuracy:%f\n", accuracy);
    end = std::chrono::high_resolution_clock::now();

    time_taken = end - start;

    printf("Time taken for test Epoch[%d]:%fSeconds\n", j + 1, time_taken.count());





    pb->stop();
    
    pbThread.join();

    delete nn;

    delete nn2;

    delete pb;

    delete initData;

    return 0;

}