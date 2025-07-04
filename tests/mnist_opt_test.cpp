#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "nn_core.h"
#include "nn_utils.h"
#include <chrono>

#include "setup_layer_info.h"

#define BUFF_SIZE 3500 //buffer size for line of mnist data
#define TRAIN_MAX 60000 //max number of lines in training set
#define TEST_MAX 10000 //max number of lines in testing set
#define NORM_FACTOR 254.0 //max value in data set for normalization
#define VAL_SIZE 784 //input layer size
#define EPOCH_MAX 3 //number epochs of training and testing 

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

int main()
{
    char* line_buff = (char*)malloc(BUFF_SIZE); 

    FILE* fdr = NULL;

    uint label;

    uint i = 0;

    uint j = 0;

    uint sz[4] = {784,32,32,10};
    eAct_func actFuncs[4] = {eAct_func::TANH, eAct_func::TANH, eAct_func::TANH, eAct_func::TANH};

    float* out = (float*)malloc(10*sizeof(float));

    float* norm_values = (float*)malloc(VAL_SIZE*sizeof(float));

    std::string linestr = "Line";

    nn_progress_bar *pb = new nn_progress_bar(linestr.c_str(),TRAIN_MAX);

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> time_taken;

    float correct_count = 0;

    float accuracy = 0.0;   

    nnInitData * initData = new nnInitData(4); 
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
    // initData->fLearningRate = 0.01f;
    //ADAM
    // initData->optParam1 = 0.9f;
    // initData->optParam2 = 0.9f;
    // initData->optParam3 = 0.1f;
    // initData->fLearningRate = 0.01f;
    //RMSPROP
    initData->optParam1 = 0.999f;
    initData->optParam2 = 0.00000001f;
    initData->fLearningRate = 0.1f;


    initData->eLossFunc = eLossFuncs::HUBER;

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