#include "nn_common_utils.h"

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