#include "cpuAccelerator.h"
#include "baseLayer.h"
#include <stdio.h>
#include <cassert>
#include <functional>

CpuAccelerator::CpuAccelerator(BaseLayer* pLayer):BaseAccelerator(pLayer)
{

    m_pLayer = pLayer;

}

CpuAccelerator::~CpuAccelerator()
{

}

void CpuAccelerator::do_forwardpass_dense_layer()
{
    BaseLayer* in_lyr = m_pLayer->GetPreviousLayer();
    BaseLayer* out_lyr = m_pLayer;

    uint in_lyr_sz = in_lyr->get_num_nodes();
    uint out_lyr_sz = out_lyr->get_num_nodes();

    uint i = 0;
    uint j = 0;

    float sigma = 0.0f;

    for(j = 0; j < out_lyr_sz; j++)
    {
        for(i = 0; i < in_lyr_sz; i++)
        {
            uint mtx_idx = (i * out_lyr_sz) + j;
            sigma += (m_pLayer->get_transform_matrix_parameter(mtx_idx) * in_lyr->get_node_value_idx(i));
                        
        }
        sigma += out_lyr->get_node_bias_idx(j);
        sigma /= in_lyr->get_num_nodes();
        // sigma = m_pActFunc->apply_act_func(sigma);
        out_lyr->set_node_value(sigma, j);
        sigma = 0.0f;
    }

    //apply activation function
    out_lyr->apply_act_func_all_nodes();
}

void CpuAccelerator::do_backwardpass_from_output_layer(std::vector<float>& pfExpOut, BaseLossFunction* lossFunc)
{
    //i = layer index
    uint j = 0; //current layer node index
    
    std::vector<float> temp(m_pLayer->get_num_nodes(), 0.0f);
    
    // m_pLossFunc->get_loss_func_derv(pfExpOut, temp);
    lossFunc->get_loss_func_derv(pfExpOut, temp);
    m_pLayer->get_delta_all_nodes(temp);

    for(j = 0; j < m_pLayer->get_num_nodes(); j++)
    {
        m_pLayer->set_node_delta(temp[j], j);
    }
    
}

void CpuAccelerator::do_backwardpass_dense_layer()
{
    //i = layer index
    uint j = 0; //current layer node index
    uint k = 0; //previous layer node index

    std::vector<float> temp(m_pLayer->get_num_nodes(), 0.0f);

    uint curr_mtx_sz = m_pLayer->get_num_nodes();
    uint nxt_mtx_sz = m_pLayer->GetNextLayer()->get_num_nodes();

    for(j = 0; j < curr_mtx_sz; j++)
    {
        // m_pLayer->GetNextLayer()
        for(k = 0; k < nxt_mtx_sz; k++)
        {
            uint mtx_idx = (j * nxt_mtx_sz) + k;
            temp[j] += m_pLayer->GetNextLayer()->get_node_delta_idx(k) * m_pLayer->GetNextLayer()->get_transform_matrix_parameter(mtx_idx);
        }
    }
    // temp *= m_pActFunc->apply_act_func_derv(m_ppLys[i]->get_node_value_idx(j));
    m_pLayer->get_delta_all_nodes(temp);
    for(j = 0; j < m_pLayer->get_num_nodes(); j++)
    {
        m_pLayer->set_node_delta(temp[j], j);
    }
}


void CpuAccelerator::do_forwardpass_conv_layer()
{

    sLayer_Parsed_Dim curr_parsed_dims = get_parsed_dims(m_pLayer->get_layer_dimensions());

    sLayer_Parsed_Dim prev_parsed_dims;

    prev_parsed_dims.num_mtx = m_pLayer->GetPreviousLayer()->get_layer_dimensions().unNoTransformParameterMtx;

    prev_parsed_dims.rows = m_pLayer->get_layer_dimensions().unInputRows;
    
    prev_parsed_dims.rows /= prev_parsed_dims.num_mtx;

    prev_parsed_dims.cols = m_pLayer->get_layer_dimensions().unInputColumns;

    uint out_mtx_size = curr_parsed_dims.rows * curr_parsed_dims.cols;

    uint kernalSz = m_pLayer->get_layer_dimensions().unTransformParametersRows * m_pLayer->get_layer_dimensions().unTransformParametersColumns;

    float val = 0.0f;

    for(uint m = 0; m < curr_parsed_dims.num_mtx; m++)
    {
        for(uint i = 0; i < curr_parsed_dims.rows; i++)
        {
            for(uint j = 0; j < curr_parsed_dims.cols; j++)
            {
                val = find_conv_at_window_all_channels(std::pair<uint, uint>(i,j), prev_parsed_dims, m);
                val += m_pLayer->get_node_bias_idx((m * out_mtx_size) + ((i * curr_parsed_dims.cols) + j));
                val /= kernalSz;
                m_pLayer->set_node_value(val, (m * out_mtx_size) + ((i * curr_parsed_dims.cols) + j));
            }
            
        }        

    }

    //apply activation function
    m_pLayer->apply_act_func_all_nodes();
}

float CpuAccelerator::find_conv_at_window_all_channels(
    std::pair<uint, uint> row_col_pos,
    sLayer_Parsed_Dim& t_prev_parsed_dims, 
    uint t_kernel_num)
{
    float val = 0.0f;
    
    for(uint m = 0; m < t_prev_parsed_dims.num_mtx; m++)
    {
        val += find_conv_at_window_one_channels(m, row_col_pos, t_prev_parsed_dims, t_kernel_num);
    }

    val /= t_prev_parsed_dims.num_mtx;

    return val;
}

float CpuAccelerator::find_conv_at_window_one_channels(
    uint mtx_num,
    std::pair<uint, uint> row_col_pos,
    sLayer_Parsed_Dim& t_prev_parsed_dims,
    uint t_kernel_num)
{
    float val = 0.0f;

    // Calculate the starting row in the previous layer for this channel
    uint absolute_row_prev = (mtx_num * t_prev_parsed_dims.rows) + row_col_pos.first;

    // Kernel size (number of elements in one kernel)
    uint kernel_rows = m_pLayer->get_layer_dimensions().unTransformParametersRows;
    uint kernel_cols = m_pLayer->get_layer_dimensions().unTransformParametersColumns;
    uint kernel_sz = kernel_rows * kernel_cols;

    // For each element in the kernel window
    for (uint i = 0; i < kernel_rows; ++i) {
        for (uint j = 0; j < kernel_cols; ++j) {
            // Index in the kernel weights for this kernel and channel
            uint kernel_idx = (t_kernel_num * kernel_sz) + (i * kernel_cols) + j;
            // Index in the input feature map for this channel
            uint input_idx = ((absolute_row_prev + i) * t_prev_parsed_dims.cols) + (row_col_pos.second + j);

            float weight = m_pLayer->get_transform_matrix_parameter(kernel_idx);
            float input = m_pLayer->GetPreviousLayer()->get_node_value_idx(input_idx);

            val += weight * input;
        }
    }

    return val;
}

void CpuAccelerator::do_backwardpass_conv_layer()
{
    sLayer_Dimensions dims = m_pLayer->GetNextLayer()->get_layer_dimensions();
    uint input_rows = dims.unInputRows;
    uint input_columns = dims.unInputColumns;
    uint filter_rows = dims.unTransformParametersRows;
    uint filter_columns = dims.unTransformParametersColumns;   

    std::vector<float> temp(m_pLayer->get_num_nodes(), 0.0f);

    uint out_rows = input_rows - filter_rows + 1;
    uint out_columns = input_columns - filter_columns + 1;    

    uint rm = 0;
    uint rn = 0;
    uint rot_filter_idx = 0;
    uint next_lyr_idx = 0;
    uint kernelSz = filter_rows * filter_columns;
    uint outSz = out_rows * out_columns;

    for(uint f  = 0; f < dims.unNoTransformParameterMtx; f++)
    {
        for (uint p = 0; p < out_rows; ++p) 
        {
            for (uint q = 0; q < out_columns; ++q) 
            {
                for (uint m = 0; m < filter_rows; ++m) 
                {
                    for (uint n = 0; n < filter_columns; ++n) 
                    {
                        // dX[p + m][q + n] += dOut[p][q] * kernel[m][n];

                        next_lyr_idx = (p * out_columns) + q;
                        next_lyr_idx += (f * outSz);
                        // compute rotated indices
                        rm = filter_rows  - 1 - m;
                        rn = filter_columns  - 1 - n;
                        rot_filter_idx = rm * filter_columns + rn;
                        rot_filter_idx += (f * kernelSz);

                        temp[((p + m) * input_columns) + (q + n)] += m_pLayer->GetNextLayer()->get_node_delta_idx(next_lyr_idx) * m_pLayer->GetNextLayer()->get_transform_matrix_parameter(rot_filter_idx);
                    }
                }
            }
        }
    }

    m_pLayer->get_delta_all_nodes(temp);
    
    for(uint j = 0; j < m_pLayer->get_num_nodes(); j++)
    {
        m_pLayer->set_node_delta(temp[j], j);
    }

}

void CpuAccelerator::do_forwardpass_pooling_layer(ePooling_type t_pooling_type)
{
    uint kernel_rows = 0;
    uint kernel_cols = 0;

    sLayer_Parsed_Dim prev_dim = m_pLayer->get_prev_layer_parsed_output_dims();

    kernel_rows = m_pLayer->get_layer_dimensions().unTransformParametersRows;
    kernel_cols = m_pLayer->get_layer_dimensions().unTransformParametersColumns;

    std::pair<uint, uint> row_col_window;
    row_col_window.first = kernel_rows;
    row_col_window.second = kernel_cols;

    std::pair<uint,uint> out_dim = find_pooling_output_dims(prev_dim, row_col_window);

    uint row_slide = out_dim.first;
    uint col_slide = out_dim.second;

    for(uint k = 0; k < prev_dim.num_mtx; k++)
    {
        for(uint i = 0; i < row_slide; i++)
        {
            for(uint j = 0; j < col_slide; j++)
            {
                std::pair<uint, uint>row_col_pos;
                row_col_pos.first = i * kernel_rows;
                row_col_pos.second = j * kernel_cols;
                float val = 0.0f;
                    
                if(ePooling_type::MAX == t_pooling_type)
                {
                    val = find_max_in_window_at(row_col_pos, prev_dim, row_col_window, k);                    
                }
                else if(ePooling_type::AVERAGE == t_pooling_type)
                {
                    val = find_avg_in_window_at(row_col_pos, prev_dim, row_col_window, k);
                }

                m_pLayer->set_node_value(val, ((k * row_slide * col_slide) + (i * col_slide) + j));
            }
        }
    }

}

uint CpuAccelerator::find_max_in_window_at(std::pair<uint, uint>row_col_pos, sLayer_Parsed_Dim prev_dim, std::pair<uint, uint>row_col_window, uint curr_mtx_idx)
{

    uint absolute_row = ((curr_mtx_idx * prev_dim.rows) + row_col_pos.first);

    float ret_max = m_pLayer->GetPreviousLayer()->get_node_value_idx((absolute_row * prev_dim.cols) + row_col_pos.second);

    for(uint i = 0; ((i < row_col_window.first) && ((i + row_col_pos.first) < prev_dim.rows)); i++)
    {
        for(uint j = 0; ((j < row_col_window.second) && ((j + row_col_pos.second) < prev_dim.cols)); j++)
        {
            float val = m_pLayer->GetPreviousLayer()->get_node_value_idx(((absolute_row + i) * prev_dim.cols) + (j + row_col_pos.second));
            if(val > ret_max)
            {
                ret_max = val;
            }
        }
    }

    return ret_max;
}

uint window_cell_count(std::pair<uint, uint>row_col_pos, sLayer_Parsed_Dim prev_dim, std::pair<uint, uint>row_col_window)
{
    uint rows = 0;
    uint cols = 0;

    if(prev_dim.rows > row_col_pos.first + row_col_window.first)
    {
        rows = row_col_window.first;
    }
    else
    {
        rows = row_col_window.first - ((row_col_pos.first + row_col_window.first) - prev_dim.rows);
    }

    if(prev_dim.cols > row_col_pos.second + row_col_window.second)
    {
        cols = row_col_window.second;
    }
    else
    {
        cols = row_col_window.second - ((row_col_pos.second + row_col_window.second) - prev_dim.cols);
    }

    return (rows * cols);

}

float CpuAccelerator::find_avg_in_window_at(std::pair<uint, uint>row_col_pos, sLayer_Parsed_Dim prev_dim, std::pair<uint, uint>row_col_window, uint curr_mtx_idx)
{

    uint cell_count = window_cell_count(row_col_pos, prev_dim, row_col_window);
    uint absolute_row = ((curr_mtx_idx * prev_dim.rows) + row_col_pos.first);

    float ret_avg = 0.0f;

    for(uint i = 0; ((i < row_col_window.first) && ((i + row_col_pos.first) < prev_dim.rows)); i++)
    {
        for(uint j = 0; ((j < row_col_window.second) && ((j + row_col_pos.second) < prev_dim.cols)); j++)
        {
            ret_avg += m_pLayer->GetPreviousLayer()->get_node_value_idx(((absolute_row + i) * prev_dim.cols) + (j + row_col_pos.second));
        }
    }

    ret_avg /= cell_count;

    return ret_avg;
}

std::vector<uint> CpuAccelerator::find_all_elements_idx_in_window(std::pair<uint, uint>row_col_pos, sLayer_Parsed_Dim prev_dim, std::pair<uint, uint>row_col_window, uint curr_mtx_idx)
{
    std::vector<uint> ret;
    uint absolute_row = ((curr_mtx_idx * prev_dim.rows) + row_col_pos.first);

    for(uint i = 0; ((i < row_col_window.first) && ((i + row_col_pos.first) < prev_dim.rows)); i++)
    {
        for(uint j = 0; ((j < row_col_window.second) && ((j + row_col_pos.second) < prev_dim.cols)); j++)
        {
            ret.push_back(((absolute_row + i) * prev_dim.cols) + (j + row_col_pos.second));
        }
    }

    return ret;

}

uint CpuAccelerator::find_max_element_idx_in_window(std::pair<uint, uint>row_col_pos, sLayer_Parsed_Dim prev_dim, std::pair<uint, uint>row_col_window, uint curr_mtx_idx)
{

    uint absolute_row = ((curr_mtx_idx * prev_dim.rows) + row_col_pos.first);

    float ret_max = m_pLayer->get_node_value_idx((absolute_row * prev_dim.cols) + row_col_pos.second);

    uint ret_idx = (absolute_row * prev_dim.cols) + row_col_pos.second;

    for(uint i = 0; ((i < row_col_window.first) && ((i + row_col_pos.first) < prev_dim.rows)); i++)
    {
        for(uint j = 0; ((j < row_col_window.second) && ((j + row_col_pos.second) < prev_dim.cols)); j++)
        {
            uint idx = ((absolute_row + i) * prev_dim.cols) + (j + row_col_pos.second);
            float val = m_pLayer->get_node_value_idx(idx);
            if(val > ret_max)
            {
                ret_max = val;
                ret_idx = idx;
            }
        }
    }

    return ret_idx;

}

void CpuAccelerator::do_backwardpass_pooling_layer(ePooling_type t_pooling_type)
{
    uint kernel_rows = 0;
    uint kernel_cols = 0;

    sLayer_Parsed_Dim prev_dim = m_pLayer->GetNextLayer()->get_prev_layer_parsed_output_dims();

    kernel_rows = m_pLayer->GetNextLayer()->get_layer_dimensions().unTransformParametersRows;
    kernel_cols = m_pLayer->GetNextLayer()->get_layer_dimensions().unTransformParametersColumns;

    std::pair<uint, uint> row_col_window;
    row_col_window.first = kernel_rows;
    row_col_window.second = kernel_cols;

    std::pair<uint,uint> out_dim = find_pooling_output_dims(prev_dim, row_col_window);

    uint row_slide = out_dim.first;
    uint col_slide = out_dim.second;

    reset_layer_deltas();

    for(uint k = 0; k < prev_dim.num_mtx; k++)
    {
        for(uint i = 0; i < row_slide; i++)
        {
            for(uint j = 0; j < col_slide; j++)
            {
                std::pair<uint, uint>row_col_pos;
                row_col_pos.first = i * kernel_rows;
                row_col_pos.second = j * kernel_cols;

                uint out_idx = ((k * row_slide * col_slide) + (i * col_slide) + j);

                if(ePooling_type::MAX == t_pooling_type)
                {
                    uint val;
                    val = find_max_element_idx_in_window(row_col_pos, prev_dim, row_col_window, k);
                    m_pLayer->set_node_delta(m_pLayer->get_node_delta_idx(val) + m_pLayer->GetNextLayer()->get_node_delta_idx(out_idx), val);                 
                }
                else if(ePooling_type::AVERAGE == t_pooling_type)
                {
                    std::vector<uint> val;
                    val = find_all_elements_idx_in_window(row_col_pos, prev_dim, row_col_window, k);
                    uint cell_count = val.size();
                    for(uint m = 0; m < cell_count; m++)
                    {
                        float delta_val = m_pLayer->get_node_delta_idx(val[m]) + (m_pLayer->GetNextLayer()->get_node_delta_idx(out_idx) / (float)cell_count);
                        m_pLayer->set_node_delta(delta_val, val[m]);
                    }
                }

            }
        }
    }

}

void CpuAccelerator::reset_layer_deltas()
{
    for(uint i = 0; i < m_pLayer->get_num_nodes(); i++)
    {
        m_pLayer->set_node_delta(0.0f, i);
    }
}

