#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "nn_core.h"
#include "nn_utils.h"
#include <chrono>

#include "nn_common_utils.h"

#define EPOCH_MAX 3 //number epochs of training and testing
#define NUMBER_TO_IDENTIFY 5
#define TEST_SAMPLE_COUNT 20


int main()
{
    char* line_buff = (char*)malloc(BUFF_SIZE); 

    FILE* fdr = NULL;

    uint label;

    uint i = 0;

    uint j = 0;

    uint sz[4] = {784,32,32,1};
    eAct_func actFuncs[4] = {eAct_func::TANH, eAct_func::TANH, eAct_func::TANH, eAct_func::TANH};

    std::vector<float> norm_values(VAL_SIZE);
    std::vector<float> out(10);

    std::string linestr = "Line";

    nn_progress_bar *pb = new nn_progress_bar(linestr.c_str(),TRAIN_MAX);

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> time_taken;

    uint u_sz = 4;

    nnInitData * initData = new nnInitData(u_sz);

    uint num_to_iden_count = 0;
    uint non_num_to_iden_count = 0;

    float * nn_output = new float[sz[3]];
    
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
        if(l == (initData->unNoLys - 1))
        {
            initData->eAct_Funcs[l] = eAct_func::SIGMOID;
        }
    }
    initData->eOpt = eOptimizers::SGD;
    initData->eLossFunc = eLossFuncs::BCE;
    initData->fLearningRate = 0.01f;

    

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

            setOutArrayBCE(label, out, NUMBER_TO_IDENTIFY);            

            //balance dataset by training 9 times over for correct label
            if(label == NUMBER_TO_IDENTIFY)
            {
                for(uint a = 0; a < 9; a++)
                {
                    nn->Train(norm_values, out);
                }
            }
            else
            {
                nn->Train(norm_values, out);
            }
            

            pb->update_progress_bar(i + 1);            

        }

        fclose(fdr);

        pb->reset();

        end = std::chrono::high_resolution_clock::now();

        time_taken = end - start;

        printf("\nTime taken for Train Epoch[%d]:%fSeconds\n", j + 1, time_taken.count());
        
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

            setOutArrayBCE(label, out, NUMBER_TO_IDENTIFY);            

            nn->Test(norm_values, out);

            pb->update_progress_bar(i + 1);
        }

        fclose(fdr);

        pb->reset();

        end = std::chrono::high_resolution_clock::now();

        time_taken = end - start;

        printf("\nTime taken for test Epoch[%d]:%fSeconds\n", j + 1, time_taken.count());
        
        fdr = fopen("MNIST/mnist_test.csv","r");
        //test for 10 NUMBER_TO_IDENTIFY and 10 non NUMBER_TO_IDENTIFY
        do
        {
            if(!fdr)
            {
                printf("fdr open fail!!!\n");
                return 0;
            }
            getNextLine(fdr, line_buff);

            label = parseLabelAndNormalizedValues(line_buff, norm_values, NORM_FACTOR);
            if(label == NUMBER_TO_IDENTIFY)
            {
                if(num_to_iden_count >= TEST_SAMPLE_COUNT)
                {
                    continue;
                }
                else
                {
                    num_to_iden_count++;
                }
            }
            else
            {
                if(non_num_to_iden_count >= TEST_SAMPLE_COUNT)
                {
                    continue;
                }
                else
                {
                    non_num_to_iden_count++;
                }
            }
            setOutArrayBCE(label, out, NUMBER_TO_IDENTIFY);
            nn->Test(norm_values, out);
            nn->Get_OutputLayer_Data(nn_output);
            printf("\nTest Sample, for label:%d output:%f\n", label, nn_output[0]);

            /* code */
        } while ((num_to_iden_count < TEST_SAMPLE_COUNT)||(non_num_to_iden_count < TEST_SAMPLE_COUNT));
        non_num_to_iden_count = 0;
        num_to_iden_count = 0;
        fclose(fdr);

    }

    pb->stop();
    
    pbThread.join();
    

    

    delete nn;

    delete pb;

    delete initData;

    return 0;

}