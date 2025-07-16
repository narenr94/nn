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
#define EPOCH_MAX 1 //number epochs of training and testing 

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

int main()
{
    char* line_buff = (char*)malloc(BUFF_SIZE); 

    FILE* fdr = NULL;

    uint label;

    uint i = 0;

    uint j = 0;

    uint sz[4] = {784,2028,32,10}; //28x28
    //26x26
    eAct_func actFuncs[4] = {eAct_func::TANH, eAct_func::TANH, eAct_func::TANH, eAct_func::TANH};

    float* out = (float*)malloc(10*sizeof(float));

    float* norm_values = (float*)malloc(VAL_SIZE*sizeof(float));

    std::string linestr = "Line";

    nn_progress_bar *pb = new nn_progress_bar(linestr.c_str(),TRAIN_MAX);

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> time_taken;

    float correct_count = 0;

    float accuracy = 0.0;  

    uint u_sz = 4;

    nnInitData * initData = new nnInitData(u_sz);
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

    // set_layer_info(sz, initData->unNoLys, initData);

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

    initData->eOpt = eOptimizers::SGD;
    initData->eLossFunc = eLossFuncs::HUBER;
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