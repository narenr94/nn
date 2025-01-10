#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "nn_core.h"
#include "nn_utils.h"
#include <chrono>

#define BUFF_SIZE 3500 //buffer size for line of mnist data
#define TRAIN_MAX 60000 //max number of lines in training set
#define TEST_MAX 10000 //max number of lines in testing set
#define NORM_FACTOR 254.0 //max value in data set for normalization
#define VAL_SIZE 784 //input layer size
#define EPOCH_MAX 3 //number epochs of training and testing
#define NUMBER_TO_IDENTIFY 5
#define TEST_SAMPLE_COUNT 20

/*
Observation
Adam doesnt seem to be working with default values for beta1, beta2 and epsilon
from trial and error found that beta1=0.9f, beta2 = 0.9f and epsilon = 0.1f works
*/

/*
getLineNumber : gets particular line from file
fdr : file descriptor to read from
line_buff : buffer tos tore output
line_num : line number to be retreived
returns : true, line found ... false if line not found
*/
bool getLineNumber(FILE* fdr, char* line_buff, uint line_num);
/*
getNextLine : gets next line from file
fdr : file descriptor to read from
line_buff : buffer tos tore output
returns : NA
*/
void getNextLine(FILE* fdr, char* line_buff);
/*
parseLabelAndNormalizedValues : parses line from mnist file and outputs normalized values and label
line_buff : buffer where mnist line is stored
norm_values : array where normalized values will be stored
norm_factor : normalization factor to be used
returns : numerical value of correct output
*/
uint parseLabelAndNormalizedValues(char* line_buff, float* norm_values, float norm_factor);
/*
setOutArray : set out array using correct label
label : value indicating correct output
out : array where expected output will be stored
returns : NA
*/
void setOutArray(uint label, float* out);

void setOutArrayBCE(uint label, float* out);

int main()
{
    char* line_buff = (char*)malloc(BUFF_SIZE); 

    FILE* fdr = NULL;

    uint label;

    uint i = 0;

    uint j = 0;

    uint sz[4] = {784,32,32,1};
    eAct_func actFuncs[4] = {eAct_func::TANH, eAct_func::TANH, eAct_func::TANH, eAct_func::TANH};

    float* out = (float*)malloc(10*sizeof(float));

    float* norm_values = (float*)malloc(VAL_SIZE*sizeof(float));

    std::string linestr = "Line";

    nn_progress_bar *pb = new nn_progress_bar(linestr.c_str(),TRAIN_MAX);

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> time_taken;

    nnInitData * initData = new nnInitData(4); 

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
    for(uint l = 0; l < initData->unNoLys; l++)
    {
        initData->unSzLys[l] = sz[l];
        
        
        initData->eAct_Funcs[l] = actFuncs[l];
        if(l == (initData->unNoLys - 1))
        {
            initData->eAct_Funcs[l] = eAct_func::SIGMOID;
        }
    }
    initData->eOpt = eOptimizers::SGD;
    initData->eLossFunc = eLossFuncs::BCE;
    initData->fLearningRate = 0.01f;

    

    NeuralNet *nn = new NeuralNet(initData);

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

            setOutArrayBCE(label, out);            

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

            setOutArrayBCE(label, out);            

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
            setOutArrayBCE(label, out);
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

void setOutArray(uint label, float* out)
{

    uint i = 0;

    for(i = 0; i < 10; i++)
    {
        if(label == i)
        {
            out[i] = 1.0;
        }
        else
        {
            out[i] = 0.0;
        }

    }
}

void setOutArrayBCE(uint label, float* out)
{

    uint i = 0;

    for(i = 0; i < 1; i++)
    {
        if(label == NUMBER_TO_IDENTIFY)
        {
            out[i] = 1.0f;
        }
        else
        {
            out[i] = 0.0f;
        }

    }
}

uint parseLabelAndNormalizedValues(char* line_buff, float* norm_values, float norm_factor)
{
    uint label = 0;

    char* token = strtok(line_buff, ",");
 
    label = (uint)atoi(token);

    uint i = 0;

    token = strtok(NULL, ",");

    while (token != NULL) {
        norm_values[i] = (float)atof(token);
        norm_values[i] /= norm_factor;
        token = strtok(NULL, ",");
        i++;
    }

    return label;
}



bool getLineNumber(FILE* fdr, char* line_buff, uint line_num)
{
    uint i = 0;
    bool ret = false;
    while(fgets(line_buff, BUFF_SIZE, fdr))
    {
        if(i == line_num)
        {
            ret = true;
            break;
        }

        i++;        
    }

    return ret;
}

void getNextLine(FILE* fdr, char* line_buff)
{
    fgets(line_buff, BUFF_SIZE, fdr);
}