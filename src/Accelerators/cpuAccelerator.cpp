#include "cpuAccelerator.h"
#include "baseLayer.h"
#include <stdio.h>

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


void CpuAccelerator::do_forwardpass_conv_layer(uint input_rows, uint input_columns, uint filter_rows, uint filter_columns)
{
    BaseLayer* in_lyr = m_pLayer->GetPreviousLayer();

    uint kernalSz = m_pLayer->get_transform_matrix_parameter_size();

    float sigma = 0.0f;

    for(uint i = 0; i < ((input_rows - filter_rows) + 1); i++)
    {
        for(uint j = 0; j < ((input_columns - filter_columns) + 1); j++)
        {
            for(uint k = 0; k < filter_rows; k++)
            {
                for(uint l = 0; l < filter_columns; l++)
                {
                    //output[i][j] += input[i + k][j + l] * filter[k][l];
                    sigma += in_lyr->get_node_value_idx(((i + k) * input_columns) + (j + l)) * m_pLayer->get_transform_matrix_parameter((k * filter_columns) + l);
                }
            }

            sigma += m_pLayer->get_node_bias_idx((i * ((input_columns - filter_columns) + 1)) + j);
            sigma /= kernalSz;
            m_pLayer->set_node_value(sigma, (i * ((input_columns - filter_columns) + 1)) + j);
            sigma = 0.0f;
        }
    }

    //apply activation function
    m_pLayer->apply_act_func_all_nodes();
}