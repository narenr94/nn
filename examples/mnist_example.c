#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include "nn_core.h"
#include <time.h>

#define BUFF_SIZE 3500 //buffer size for line of mnist data
#define TRAIN_MAX 60000 //max number of lines in training set
#define TEST_MAX 10000 //max number of lines in testing set
#define NORM_FACTOR 254.0 //max value in data set for normalization
#define VAL_SIZE 784 //input layer size
#define EPOCH_MAX 5 //number epochs of training and testing 

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

    float* out = (float*)malloc(10*sizeof(float));

    float* norm_values = (float*)malloc(VAL_SIZE*sizeof(float));

    NeuralNet *nn = new NeuralNet();

    string linestr = "Line";

    nn_progress_bar *pb = new nn_progress_bar(linestr.c_str(),TRAIN_MAX);

    clock_t t;

    double time_taken;

    float correct_count = 0;

    float accuracy = 0.0;    

    nn->init(4, sz, SIGMOID, eLOGLEVEL_ERROR, false, 0.5);

    nn->populateWeightsAndBiasesWithRandomNumbers();

    std::thread pbThread(&nn_progress_bar::print_progress_bar_periodic, pb, 0, 1000);

    for(j = 0; j < EPOCH_MAX; j++)
    {
        t = clock();

        printf("Epoch[%d] Started!!!\n", j + 1);
        NNLOG_MIL("Epoch[%d] Started!!!\n", j + 1);

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
        NNLOG_MIL("Train Accuracy:%f\n", accuracy);
        t = clock() - t;

        time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

        printf("Time taken for Train Epoch[%d]:%fSeconds\n", j + 1, time_taken);
        NNLOG_MIL("Time taken for Train Epoch[%d]:%fSeconds\n", j + 1,time_taken);
        
        correct_count = 0.0;

        pb->setMax(TEST_MAX);

        t = clock();

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
        NNLOG_MIL("Test Accuracy:%f\n", accuracy);
        t = clock() - t;

        time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

        printf("Time taken for test Epoch[%d]:%fSeconds\n", j + 1, time_taken);
        NNLOG_MIL("Time taken for test Epoch[%d]:%fSeconds\n", j + 1,time_taken);
        
        correct_count = 0.0;

    }

    pb->stop();
    
    pbThread.join();
    

    printf("start dump\n");
    t = clock();

    nn->dump_nn();

    t = clock() - t;

    time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds

    printf("Time taken to dump file:%fSeconds\n", time_taken);
    NNLOG_MIL("Time taken to dump file:%fSeconds\n", time_taken);

    printf("done dump\n");

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