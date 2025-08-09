#include "nn_defines.h"
#include <cassert>
#include <fstream>

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
    layer_dimensions.emplace_back(0, 0, 1, InLyrSz, 0, 0, 1);
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

void nnInitData::add_conv_layer(uint t_in_rows, uint t_in_cols, eConvKernelSize t_kernel_size, uint t_num_kernels, eAct_func t_act_func, float t_act_param)
{
    assert((t_in_rows * t_in_cols) == (layer_dimensions[unNoLys - 1].unOutputColumns * layer_dimensions[unNoLys - 1].unOutputRows));
    unNoLys++;
    uint kernelRow;
    uint kernelColumn;
    uint outRow;
    uint outCols;

    switch(t_kernel_size)
    {
        case eConvKernelSize::Sz3x3:
            kernelRow = 3;
            kernelColumn = 3;
            break;
        case eConvKernelSize::Sz5x5:
            kernelRow = 5;
            kernelColumn = 5;
            break;
        case eConvKernelSize::Sz7x7:
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

void nnInitData::add_pooling_layer(ePooling_type type, ePoolingKernelSize krSz, uint stride)
{
    unNoLys++;

    sLayer_Parsed_Dim prev_dim = get_parsed_dims(layer_dimensions[unNoLys - 2]);
    std::pair<uint,uint> kr_row_col = get_pooling_window_rows_cols(krSz, prev_dim);
    std::pair<uint,uint> out_dim = find_pooling_output_dims(prev_dim, kr_row_col);

    layer_dimensions.emplace_back(layer_dimensions[unNoLys - 2].unOutputRows, layer_dimensions[unNoLys - 2].unOutputColumns, out_dim.first * layer_dimensions[unNoLys - 2].unNoTransformParameterMtx, out_dim.second, kr_row_col.first, kr_row_col.second, layer_dimensions[unNoLys - 2].unNoTransformParameterMtx);
    eAct_Funcs.emplace_back(eAct_func::RELU);
    e_layer_type.emplace_back(eLayer_type::POOLING);
    actParam1.emplace_back(stride);

}

std::vector<std::string> split_by_lines(const std::string& block)
{
    std::vector<std::string> lines;
    std::istringstream ss(block);
    std::string line;

    while (std::getline(ss, line)) {
        // Optional: Trim whitespace or skip empty lines
        if (!line.empty())
            lines.push_back(line);
    }

    return lines;
}

std::pair<std::string, std::vector<std::string>> parse_line(const std::string& line)
{
    std::istringstream iss(line);
    std::string tag;
    std::vector<std::string> values;

    if (iss >> tag) {
        std::string value;
        while (iss >> value) {
            values.push_back(value);
        }
    }

    return {tag, values};
}

std::string read_file(const std::string& fileName)
{
    std::ifstream in(fileName);
    assert(in);
    std::ostringstream ss;
    ss << in.rdbuf(); // Reads full content
    return ss.str();
}

void save_to_file(const std::string& data, const std::string& filename)
{
    std::ofstream out(filename);
    assert(out);
    out << data;
    out.close();
}

std::vector<std::string> split_by_delimiter(const std::string& data, const std::string& delimiter)
{
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = data.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(data.substr(start, end - start));
        start = end + delimiter.length();
        end = data.find(delimiter, start);
    }

    // Add last segment
    parts.push_back(data.substr(start));

    return parts;
}

std::pair<uint,uint> get_pooling_window_rows_cols(ePoolingKernelSize t_pooling_kernel_sz, sLayer_Parsed_Dim& prev_dim)
{
    std::pair<uint,uint> ret;

    switch(t_pooling_kernel_sz)
    {
        case ePoolingKernelSize::Sz2x2:
            ret.first = 2;
            ret.second = 2;
            break;
        case ePoolingKernelSize::Sz3x3:
            ret.first = 3;
            ret.second = 3;
            break;
        case ePoolingKernelSize::GLOBAL:
            ret.first = prev_dim.rows / prev_dim.num_mtx;
            ret.second = prev_dim.cols;
            break;
        default:
            assert(0);
            break;
    }

    return ret;
}

std::pair<uint, uint> find_pooling_output_dims(sLayer_Parsed_Dim& in_dims, std::pair<uint,uint> krSz)
{
    uint col_slide = in_dims.cols/krSz.second;
    if(in_dims.cols%krSz.second)
    {
        col_slide += 1;
    }

    uint row_slide = in_dims.rows/krSz.first;
    if(in_dims.rows%krSz.first)
    {
        row_slide += 1;
    }

    return std::pair<uint, uint>(row_slide, col_slide);

}

sLayer_Parsed_Dim get_parsed_dims(const sLayer_Dimensions& dims)
{
    sLayer_Parsed_Dim ret_dims;

    ret_dims.num_mtx = dims.unNoTransformParameterMtx;
    ret_dims.rows = dims.unOutputRows;

    if(ret_dims.num_mtx == 0)//incase of prev layer = input layer
    {
        ret_dims.num_mtx = 1;
    }

    ret_dims.rows /= ret_dims.num_mtx;

    ret_dims.cols = dims.unOutputColumns;

    return ret_dims;
}
