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
    char* line_buff = (char*)malloc(BUFF_SIZE); 

    FILE* fdr = NULL;

    uint label;

    uint i = 0;

    uint j = 0;

    uint sz[5] = {784,32,32,10};

    // eAct_func actFuncs[4] = {eAct_func::TANH, eAct_func::TANH, eAct_func::TANH, eAct_func::TANH};

    std::vector<float> norm_values(VAL_SIZE);

    std::vector<float> out(10);

    std::string linestr = "Line";

    nn_progress_bar *pb = new nn_progress_bar(linestr.c_str(),TRAIN_MAX);

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> time_taken;
    
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

        nn->populateWeightsAndBiasesWithRandomNumbers();

        std::thread pbThread(&nn_progress_bar::print_progress_bar_periodic, pb, 0, 1000);

        for(j = 0; j < EPOCH_MAX; j++)
        {
            pb->reset();

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

        pb->stop();
        
        pbThread.join();

        delete nn;

        delete initData;
    }


    delete pb;

   

    return 0;

}