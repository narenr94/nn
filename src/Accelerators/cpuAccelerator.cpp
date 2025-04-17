#include "cpuAccelerator.h"
#include "nn_layer.h"
#include "nn_l2l_weight_matrix.h"

CpuAccelerator::CpuAccelerator(nn_layer* pLayer):BaseAccelerator(pLayer)
{

    m_pLayer = pLayer;

}

CpuAccelerator::~CpuAccelerator()
{

}

void CpuAccelerator::do_forwardpass_dense_layer()
{
    // nn_layer* in_lyr = m_pNN->GetLayer(unInLayerIdx - 1);
    // nn_layer* out_lyr = m_pNN->GetLayer(unInLayerIdx);

    nn_layer* in_lyr = m_pLayer->GetPreviousLayer();
    nn_layer* out_lyr = m_pLayer;

    nn_l2l_weight_matrix* curr_mtx_ptr = m_pLayer->GetWeightMatrix();

    uint in_lyr_sz = in_lyr->get_num_nodes();
    uint out_lyr_sz = out_lyr->get_num_nodes();

    uint i = 0;
    uint j = 0;

    float sigma = 0;

    for(j = 0; j < out_lyr_sz; j++)
    {
        for(i = 0; i < in_lyr_sz; i++)
        {
            sigma += (curr_mtx_ptr->get_weight(i, j) * in_lyr->get_node_value_idx(i));
                        
        }
        sigma += out_lyr->get_node_bias_idx(j);
        sigma /= in_lyr->get_num_nodes();
        // sigma = m_pActFunc->apply_act_func(sigma);
        out_lyr->set_node_value(sigma, j);
        sigma = 0;
    }

    //apply activation function
    out_lyr->apply_act_func_all_nodes();
}

void CpuAccelerator::do_backwardpass_dense_layer_output_layer(float* pfExpOut, BaseLossFunction* lossFunc)
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

    for(j = 0; j < m_pLayer->get_num_nodes(); j++)
    {
        temp[j] = 0.0f;

        // m_pLayer->GetNextLayer()
        for(k = 0; k < m_pLayer->GetNextLayer()->get_num_nodes(); k++)
        {
            temp[j] += m_pLayer->GetNextLayer()->get_node_delta_idx(k) * m_pLayer->GetWeightMatrix()->get_weight(j, k);
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