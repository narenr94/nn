#include "nn_common_utils.h"
#include "nn_utils.h"

#include <stdlib.h>
#include <string.h>

void setOutArray(uint label, std::vector<float>& out)
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

uint parseLabelAndNormalizedValues(char* line_buff, std::vector<float>& norm_values, float norm_factor)
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

void set_layer_info(uint* sz, uint sz_sz, nnInitData * initData)
{
    int i = 0;

    //input layer
    initData->layer_dimensions[i].unInputColumns = 0;
    initData->layer_dimensions[i].unInputRows = 0;
    initData->layer_dimensions[i].unNoTransformParameterMtx = 1;
    initData->layer_dimensions[i].unOutputColumns = sz[0];
    initData->layer_dimensions[i].unOutputRows = 1;
    initData->layer_dimensions[i].unTransformParametersColumns = 0;
    initData->layer_dimensions[i].unTransformParametersRows = 0;

    for(i = 1; i < sz_sz; i++)
    {
        //hidden layer and output layer
        initData->layer_dimensions[i].unInputColumns = sz[i - 1];
        initData->layer_dimensions[i].unInputRows = 1;
        initData->layer_dimensions[i].unNoTransformParameterMtx = 1;
        initData->layer_dimensions[i].unOutputColumns = sz[i];
        initData->layer_dimensions[i].unOutputRows = 1;
        initData->layer_dimensions[i].unTransformParametersColumns = sz[i];
        initData->layer_dimensions[i].unTransformParametersRows = sz[i - 1];
    }
}

void setOutArrayBCE(uint label, std::vector<float> out, uint num_identify)
{

    uint i = 0;

    for(i = 0; i < 1; i++)
    {
        if(label == num_identify)
        {
            out[i] = 1.0f;
        }
        else
        {
            out[i] = 0.0f;
        }

    }
}

void Init_NN_for_MNIST(NeuralNet* nn)
{
    nn->populateWeightsAndBiasesWithRandomNumbers();
}

void Train_NN_for_MNIST(NeuralNet* nn)
{
    float accuracy = 0.0;
    uint correct_count = 0;
    uint label;

    char* line_buff = (char*)malloc(BUFF_SIZE); 

    FILE* fdr = NULL;

    std::vector<float> norm_values(VAL_SIZE);
    std::vector<float> out(10);

    std::string linestr = "Line";
    nn_progress_bar *pb = new nn_progress_bar(linestr.c_str(),TRAIN_MAX);
    std::thread pbThread([&]() {
        pb->print_progress_bar_periodic();
    });

    pb->setMax(TRAIN_MAX);
    fdr = fopen("MNIST/mnist_train.csv","r");
    //getNextLine(fdr, line_buff);

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> time_taken;
    start = std::chrono::high_resolution_clock::now();

    for(uint i = 0; i < TRAIN_MAX; i++)
    {
        
        if(!fdr)
        {
            printf("fdr open fail!!!\n");
            return;
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

    pb->stop();    
    pbThread.join();
    delete pb;

    accuracy = correct_count / ((float)TRAIN_MAX);

    printf("\nTrain Accuracy:%f\n", accuracy);
    end = std::chrono::high_resolution_clock::now();

    time_taken = end - start;

    printf("Time taken for Train:%fSeconds\n", time_taken.count());

    
    delete [] line_buff;

}

void Test_NN_for_MNIST(NeuralNet* nn)
{
    float accuracy = 0.0;
    uint correct_count = 0;
    uint label;

    char* line_buff = (char*)malloc(BUFF_SIZE); 

    

    std::vector<float> norm_values(VAL_SIZE);
    std::vector<float> out(10);

    std::string linestr = "Line";
    nn_progress_bar *pb = new nn_progress_bar(linestr.c_str(),TEST_MAX);
    std::thread pbThread([&]() {
        pb->print_progress_bar_periodic();
    });
    pb->setMax(TEST_MAX);

    FILE* fdr = NULL;
    fdr = fopen("MNIST/mnist_test.csv","r");

    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> time_taken;
    start = std::chrono::high_resolution_clock::now();

    

    //getNextLine(fdr, line_buff);

    for(uint i = 0; i < TEST_MAX; i++)
    {
        
        if(!fdr)
        {
            printf("fdr open fail!!!\n");
            return;
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

    pb->stop();    
    pbThread.join();
    delete pb;

    accuracy = correct_count / ((float)TEST_MAX);

    printf("\nTest Accuracy:%f\n", accuracy);
    end = std::chrono::high_resolution_clock::now();

    time_taken = end - start;

    printf("Time taken for test:%fSeconds\n", time_taken.count());    
    
    delete [] line_buff;
}

