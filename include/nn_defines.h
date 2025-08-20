#ifndef NN_DEFINES_H
#define NN_DEFINES_H

#include "nn_math.h"

#include <vector>
#include <string>
#include <sstream>
#include <map>


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

extern std::map<eAct_func, std::string> actFuncToString;

extern std::map<std::string, eAct_func> stringToActFunc;

/*
    list of layer type
    Note: keep DENSE at last to keep test scripts intact
*/
enum eLayer_type{
    INPUT,
    CONV,
    POOLING,
    DENSE
};

extern std::map<eLayer_type, std::string> layerTypeToString;

extern std::map<std::string, eLayer_type> stringToLayerType;

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

extern std::map<eLossFuncs, std::string> lossFuncToString;

extern std::map<std::string, eLossFuncs> stringToLossFunc;

enum ePooling_type{
    AVERAGE,
    MAX,
    NA // not applicable - for non-pooling layer
};

extern std::map<ePooling_type, std::string> poolingTypeToString;

extern std::map<std::string, ePooling_type> stringToPoolingType;


/*
    list of activation functions
    Note : keep ADAM in bottom to keep tests intact
*/
enum eOptimizers{
    SGD,
    RMSPROP,
    ADAM
};

extern std::map<eOptimizers, std::string> optimizerToString;

extern std::map<std::string, eOptimizers> stringToOptimizer;


enum eConvKernelSize{
    Sz3x3,
    Sz5x5,
    Sz7x7
};

enum ePoolingKernelSize{
    KrSz3x3,
    KrSz2x2,
    GLOBAL
};


struct sMtx_Dim{

    uint rows = 0;
    uint columns = 0;

};

#define DIMS_SIZE 7 //number of members in sLayer_Dimensions
struct sLayer_Dimensions{

    uint unInputRows;
    uint unInputColumns;
    uint unOutputRows;
    uint unOutputColumns;
    uint unTransformParametersRows;
    uint unTransformParametersColumns;
    uint unNoTransformParameterMtx;

    sLayer_Dimensions();

    sLayer_Dimensions(const sLayer_Dimensions& other);

    sLayer_Dimensions(sMtx_Dim in_dim, sMtx_Dim out_dim, sMtx_Dim tran_dim, uint no_trans_mtx);
};



struct sLayer_Parsed_Dim{

    uint rows;
    uint cols;
    uint num_mtx;

};

//when below changed make sure to update get set save and load in nn_core
struct nnInitData{

    //for debug
    uint ID;
    
    //layer stuff
    std::vector<sLayer_Dimensions> layer_dimensions;
    std::vector<eAct_func> eAct_Funcs;
    std::vector<eLayer_type> e_layer_type;
    std::vector<float>actParam1; //LEAKY_RELU : delta
    std::vector<ePooling_type>ePoolingType;
    uint unNoLys = 0;

    //entire network stuff
    float fLearningRate = 0.5f;
    eOptimizers eOpt = eOptimizers::SGD;
    float optParam[3] = {0.0f, 0.0f, 0.0f}; 
    //optParam[0]RMS_PROP : beta, ADAM : beta1
    //optParam[1]RMS_PROP : epsilon, ADAM : beta2
    //optParam[2]ADAM : epsilon    
    eLossFuncs eLossFunc = eLossFuncs::MSE;
    float lossParam = 0.0f; //HUBER : delta

    nnInitData()
    {}

    nnInitData(uint sz);

    nnInitData(uint InLyrSz, eOptimizers t_opt, std::vector<float> t_optParam, eLossFuncs t_loss_func, float t_loss_param, float t_learning_rate);

    nnInitData(const nnInitData& other);

    ~nnInitData()
    {
    }

    void add_dense_layer(uint OutSz, eAct_func t_act_func, float t_act_param);

    void add_conv_layer(sMtx_Dim in_dim, eConvKernelSize t_kernel_size, uint t_num_kernels, eAct_func t_act_func, float t_act_param);

    void add_pooling_layer(sMtx_Dim in_dim, ePooling_type type, ePoolingKernelSize krSz);

};

struct sNN_General_Data{
    uint m_unNumLys;
    float m_fLearningRate;
    eOptimizers m_eOpt;
    eLossFuncs m_eLossFunc;
    float m_optParam[3];
    float m_lossParam;
    std::vector<eLayer_type> layer_types;
};

std::vector<std::string> split_by_lines(const std::string& block);

std::pair<std::string, std::vector<std::string>> parse_line(const std::string& line);

std::string read_file(const std::string& fileName);

void save_to_file(const std::string& data, const std::string& filename);

std::vector<std::string> split_by_delimiter(const std::string& data, const std::string& delimiter = "***");

std::pair<uint,uint> get_pooling_window_rows_cols(ePoolingKernelSize t_pooling_kernel_sz, sLayer_Parsed_Dim& prev_dim);

std::pair<uint, uint> find_pooling_output_dims(sLayer_Parsed_Dim& in_dims, std::pair<uint,uint> krSz);

sLayer_Parsed_Dim get_parsed_dims(const sLayer_Dimensions& dims);

#endif