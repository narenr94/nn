#include "nn_defines.h"
#include <cassert>

sLayer_Dimensions::sLayer_Dimensions()
{
    unInputRows;
    unInputColumns;
    unOutputRows;
    unOutputColumns;
    unTransformParametersRows;
    unTransformParametersColumns;
    unNoTransformParameterMtx;
}

sLayer_Dimensions::sLayer_Dimensions(const sLayer_Dimensions& other)
{
    unInputRows = other.unInputRows;
    unInputColumns = other.unInputColumns;
    unOutputRows = other.unOutputRows;
    unOutputColumns = other.unOutputColumns;
    unTransformParametersRows = other.unTransformParametersRows;
    unTransformParametersColumns = other.unTransformParametersColumns;
    unNoTransformParameterMtx = other.unNoTransformParameterMtx;
}

sLayer_Dimensions::sLayer_Dimensions(uint in_rows, uint in_columns, uint out_rows, uint out_cols, uint trans_rows, uint trans_cols, uint no_trans_mtx)
{
    unInputRows = in_rows;
    unInputColumns = in_columns;
    unOutputRows = out_rows;
    unOutputColumns = out_cols;
    unTransformParametersRows = trans_rows;
    unTransformParametersColumns = trans_cols;
    unNoTransformParameterMtx = no_trans_mtx;
}

nnInitData::nnInitData(uint sz)
{
    layer_dimensions.reserve(sz);
    eAct_Funcs.reserve(sz);
    e_layer_type.reserve(sz);
    actParam1.reserve(sz);
}

nnInitData::nnInitData(uint InLyrSz, eOptimizers t_opt, std::vector<float> t_optParam, eLossFuncs t_loss_func, float t_loss_param, float t_learning_rate)
{
    assert(t_optParam.size() == 3); //need three params for optimizer
    unNoLys++;
    layer_dimensions.emplace_back(0, 0, 1, InLyrSz, 0, 0, 0);
    eAct_Funcs.emplace_back(eAct_func::TANH); //doesnt matter
    e_layer_type.emplace_back(eLayer_type::INPUT);
    actParam1.emplace_back(0.0f); //doesnt matter

    fLearningRate = t_learning_rate;

    eOpt = t_opt;
    optParam[0] = t_optParam[0];
    optParam[1] = t_optParam[1];
    optParam[2] = t_optParam[2];

    eLossFunc = t_loss_func;
    lossParam = t_loss_param;

}

nnInitData::nnInitData(const nnInitData& other)
{
    unNoLys = other.unNoLys;
    fLearningRate = other.fLearningRate;
    eOpt = other.eOpt;
    optParam[0] = other.optParam[0];
    optParam[1] = other.optParam[1];
    optParam[2] = other.optParam[2];
    eLossFunc = other.eLossFunc;
    lossParam = other.lossParam;

    for(uint i = 0; i < unNoLys; i++)
    {
        layer_dimensions.emplace_back(other.layer_dimensions[i]);
        eAct_Funcs.emplace_back(other.eAct_Funcs[i]);
        e_layer_type.emplace_back(other.e_layer_type[i]);
        actParam1.emplace_back(other.actParam1[i]);
    }
}

void nnInitData::add_dense_layer(uint OutSz, eAct_func t_act_func, float t_act_param)
{
    uint InSz = layer_dimensions[unNoLys - 1].unOutputColumns * layer_dimensions[unNoLys - 1].unOutputRows;

    unNoLys++;
    layer_dimensions.emplace_back(1, InSz, 1, OutSz, InSz, OutSz, 1);
    eAct_Funcs.emplace_back(t_act_func);
    e_layer_type.emplace_back(eLayer_type::DENSE);
    actParam1.emplace_back(t_act_param);
}

void nnInitData::add_conv_layer(uint t_in_rows, uint t_in_cols, eKernelSize t_kernel_size, uint t_num_kernels, eAct_func t_act_func, float t_act_param)
{
    assert((t_in_rows * t_in_cols) == (layer_dimensions[unNoLys - 1].unOutputColumns * layer_dimensions[unNoLys - 1].unOutputRows));
    unNoLys++;
    uint kernelRow;
    uint kernelColumn;
    uint outRow;
    uint outCols;

    switch(t_kernel_size)
    {
        case eKernelSize::Sz3x3:
            kernelRow = 3;
            kernelColumn = 3;
            break;
        case eKernelSize::Sz5x5:
            kernelRow = 5;
            kernelColumn = 5;
            break;
        case eKernelSize::Sz7x7:
            kernelRow = 7;
            kernelColumn = 7;
            break;
        default:
            kernelRow = 3;
            kernelColumn = 3;
            break;
    }

    outRow = (t_in_rows - kernelRow + 1) * t_num_kernels;
    outCols = t_in_cols - kernelColumn + 1;

    layer_dimensions.emplace_back(t_in_rows, t_in_cols, outRow, outCols, kernelRow, kernelColumn, t_num_kernels);
    eAct_Funcs.emplace_back(t_act_func);
    e_layer_type.emplace_back(eLayer_type::CONV);
    actParam1.emplace_back(t_act_param);

}

