#ifndef NN_DEFINES_H
#define NN_DEFINES_H

#include "nn_math.h"
#include <vector>

/*
    list of activation functions
    Note: keep TANH at last to keep test scripts intact
*/
enum eAct_func{
    SIGMOID,
    RELU,
    LEAKY_RELU,
    SOFTMAX,
    TANH    
};

/*
    list of layer type
    Note: keep DENSE at last to keep test scripts intact
*/
enum eLayer_type{
    INPUT,
    CONV,
    DENSE
};

struct sLayer_Dimensions{

    uint unInputRows;
    uint unInputColumns;
    uint unOutputRows;
    uint unOutputColumns;
    uint unTransformParametersRows;
    uint unTransformParametersColumns;
    uint unNoTransformParameterMtx;

    sLayer_Dimensions()
    {
        unInputRows;
        unInputColumns;
        unOutputRows;
        unOutputColumns;
        unTransformParametersRows;
        unTransformParametersColumns;
        unNoTransformParameterMtx;
    }

    sLayer_Dimensions(uint in_rows, uint in_columns, uint out_rows, uint out_cols, uint trans_rows, uint trans_cols, uint no_trans_mtx)
    {
        unInputRows = in_rows;
        unInputColumns = in_columns;
        unOutputRows = out_rows;
        unOutputColumns = out_cols;
        unTransformParametersRows = trans_rows;
        unTransformParametersColumns = trans_cols;
        unNoTransformParameterMtx = no_trans_mtx;
    }

};


/*
    list of activation functions
    Note : keep CCE in bottom to keep tests intact
*/
enum eLossFuncs{
    MSE, //mean squared error
    MAE, //mean absolute error
    HUBER, //huber loss
    BCE, //binary cross entropy loss
    CCE //competitive cross entropy loss
};




/*
    list of activation functions
    Note : keep ADAM in bottom to keep tests intact
*/
enum eOptimizers{
    SGD,
    RMSPROP,
    ADAM
};


//when below changed make sure to update get set save and load in nn_core
struct nnInitData{

    uint unNoLys = 0;
    std::vector<sLayer_Dimensions> layer_dimensions;
    std::vector<eAct_func> eAct_Funcs;
    std::vector<eLayer_type> e_layer_type;
    float fLearningRate = 0.5f;
    uint ID = 0;
    eOptimizers eOpt = eOptimizers::SGD;
    eLossFuncs eLossFunc = eLossFuncs::MSE;
    float optParam1 = 0.0f; //RMS_PROP : beta, ADAM : beta1
    float optParam2 = 0.0f; //RMS_PROP : epsilon, ADAM : beta2
    float optParam3 = 0.0f; //ADAM : epsilon
    std::vector<float>actParam1; //LEAKY_RELU : delta
    float lossParam1 = 0.0f; //HUBER : delta
    //ToDo: parameters for actFunc and Optimizers
    
    nnInitData(uint NumLys)
    {
        unNoLys = NumLys;
        layer_dimensions.resize(unNoLys);
        eAct_Funcs.resize(unNoLys);
        e_layer_type.resize(unNoLys);
        actParam1.resize(unNoLys);

        for(uint i = 0; i < NumLys; i++)
        {
            eAct_Funcs[i] = eAct_func::TANH;
            e_layer_type[i] = eLayer_type::DENSE;
            actParam1[i] = 0.0f;
        }
    };

    ~nnInitData()
    {
    }

};

#endif