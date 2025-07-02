#ifndef SET_LAYER_INFO_H
#define SET_LAYER_INFO_H

#include "nn_math.h"
#include "nn_core.h"

void set_layer_info(uint* sz, uint sz_sz, nnInitData * initData)
{
    int i = 0;

    //input layer
    initData->layer_dimensions[i].unInputColumns = 0;
    initData->layer_dimensions[i].unInputRows = 0;
    initData->layer_dimensions[i].unNoTransformParameterMtx = 0;
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

#endif