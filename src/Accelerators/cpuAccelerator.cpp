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

void CpuAccelerator::do_backwardpass_from_output_layer(float* pfExpOut, BaseLossFunction* lossFunc)
{
    //i = layer index
    uint j = 0; //current layer node index
    
    float * temp = new float[m_pLayer->get_num_nodes()];
    
    // m_pLossFunc->get_loss_func_derv(pfExpOut, temp);
    lossFunc->get_loss_func_derv(pfExpOut, temp);
    m_pLayer->get_delta_all_nodes(temp);

    for(j = 0; j < m_pLayer->get_num_nodes(); j++)
    {
        m_pLayer->set_node_delta(temp[j], j);
    }

    delete [] temp;
    
}

void CpuAccelerator::do_backwardpass_dense_layer()
{
    //i = layer index
    uint j = 0; //current layer node index
    uint k = 0; //previous layer node index

    float * temp = new float[m_pLayer->get_num_nodes()];

    uint curr_mtx_sz = m_pLayer->get_num_nodes();
    uint nxt_mtx_sz = m_pLayer->GetNextLayer()->get_num_nodes();

    for(j = 0; j < curr_mtx_sz; j++)
    {
        temp[j] = 0.0f;

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

    delete [] temp;
}


void CpuAccelerator::do_forwardpass_conv_layer(sLayer_Dimensions t_dims)
{
    uint input_rows = t_dims.unInputRows;
    uint input_columns = t_dims.unInputColumns;
    uint filter_rows = t_dims.unTransformParametersRows;
    uint filter_columns = t_dims.unTransformParametersColumns;
    uint output_rows = t_dims.unInputRows - t_dims.unTransformParametersRows + 1; //per filter
    uint output_columns = t_dims.unInputColumns - t_dims.unTransformParametersColumns + 1; //per filter
    

    BaseLayer* in_lyr = m_pLayer->GetPreviousLayer();

    uint kernalSz = filter_rows * filter_columns;

    uint outSz = output_rows * output_columns;

    float sigma = 0.0f;

    for(uint f = 0; f < t_dims.unNoTransformParameterMtx; f++)
    {

        for(uint i = 0; i < output_rows; i++)
        {
            for(uint j = 0; j < output_columns; j++)
            {
                for(uint k = 0; k < filter_rows; k++)
                {
                    for(uint l = 0; l < filter_columns; l++)
                    {
                        //output[i][j] += input[i + k][j + l] * filter[k][l];
                        sigma += in_lyr->get_node_value_idx(((i + k) * input_columns) + (j + l)) * m_pLayer->get_transform_matrix_parameter(((k * filter_columns) + l) + (f * kernalSz));
                    }
                }

                sigma += m_pLayer->get_node_bias_idx(((i * output_columns) + j) + (f * outSz));
                sigma /= kernalSz;
                m_pLayer->set_node_value(sigma, ((i * output_columns) + j) + (f * outSz));
                sigma = 0.0f;
            }
        }
    }

    //apply activation function
    m_pLayer->apply_act_func_all_nodes();
}

void CpuAccelerator::do_backwardpass_conv_layer(sLayer_Dimensions t_dims)
{
    uint input_rows = t_dims.unInputRows;
    uint input_columns = t_dims.unInputColumns;
    uint filter_rows = t_dims.unTransformParametersRows;
    uint filter_columns = t_dims.unTransformParametersColumns;

    uint kernelSz = filter_rows * filter_columns;
   

    float * temp = new float[m_pLayer->get_num_nodes()];

    for(uint i = 0; i < m_pLayer->get_num_nodes(); i++)
    {
        temp[i] = 0.0f;
    }

    uint out_rows = input_rows - filter_rows + 1;
    uint out_columns = input_columns - filter_columns + 1;

    uint outSz = out_rows * out_columns;

    uint p = 0; //out_rows
    uint q = 0; //out_columns

    uint m = 0; //filter_rows
    uint n = 0; //filter_columns

    uint rm = 0;
    uint rn = 0;
    uint rot_filter_idx = 0;


    float next_lyr_idx = 0.0f;
    float filter_idx = 0.0f;

    for(uint f  = 0; f < t_dims.unNoTransformParameterMtx; f++)
    {

        for (p = 0; p < out_rows; ++p) 
        {
            for (q = 0; q < out_columns; ++q) 
            {
                for (m = 0; m < filter_rows; ++m) 
                {
                    for (n = 0; n < filter_columns; ++n) 
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

    delete [] temp;
}

void CpuAccelerator::do_forwardpass_pooling_layer(ePooling_type t_pooling_type, ePoolingKernelSize t_pooling_kernel_sz, uint t_stride)
{
    uint kernel_rows = 0;
    uint kernel_cols = 0;

    sLayer_Parsed_Dim prev_dim = m_pLayer->get_prev_layer_parsed_output_dims();

    std::pair<uint,uint> row_col_window = get_pooling_window_rows_cols(t_pooling_kernel_sz, prev_dim);
    kernel_rows = row_col_window.first;
    kernel_cols = row_col_window.second;

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
                uint val = 0.0f;
                    
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

void CpuAccelerator::do_backwardpass_pooling_layer(ePooling_type t_pooling_type, ePoolingKernelSize t_pooling_kernel_sz, uint t_stride)
{
    uint kernel_rows = 0;
    uint kernel_cols = 0;

    sLayer_Parsed_Dim prev_dim = m_pLayer->get_prev_layer_parsed_output_dims();

    std::pair<uint,uint> row_col_window = get_pooling_window_rows_cols(t_pooling_kernel_sz, prev_dim);
    kernel_rows = row_col_window.first;
    kernel_cols = row_col_window.second;

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
                uint val = 0.0f;
                    
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