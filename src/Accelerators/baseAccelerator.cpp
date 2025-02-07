#include "baseAccelerator.h"
#include "nn_core.h"

BaseAccelerator::BaseAccelerator(NeuralNet* pNN)
{

    m_pNN = pNN;

}

BaseAccelerator::~BaseAccelerator()
{

}

void BaseAccelerator::do_forwardpass_to_next_layer(uint unInLayerIdx)
{
    nn_layer* in_lyr = m_pNN->GetLayer(unInLayerIdx);
    nn_layer* out_lyr = m_pNN->GetLayer(unInLayerIdx + 1);

    nn_l2l_weight_matrix* curr_mtx_ptr = m_pNN->GetMatrix(unInLayerIdx);

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

void BaseAccelerator::find_delta_of_all_nodes(float* pfExpOut)
{
    uint i = 0; //layer index
    uint j = 0; //current layer node index
    uint k = 0; //previous layer node index

    // float* error = new float[m_ppLys[m_unNumLys - 1]->get_num_nodes()];

    // float total_error = calculate_error(pfExpOut, error);

    uint numLys = m_pNN->GetNumLys();

    //find correct pred for softmax
    for(i = (numLys - 1); i > INPUT_LAYER_ID; i--)
    {
        // float * temp = new float[m_ppLys[i]->get_num_nodes()];
        float * temp = new float[m_pNN->GetLayer(i)->get_num_nodes()];
        if(m_pNN->GetLayer(i)->get_layer_type() == OUTPUT_LYR)//output layer
        {
            // m_pLossFunc->get_loss_func_derv(pfExpOut, temp);
            m_pNN->GetLossFunc()->get_loss_func_derv(pfExpOut, temp);
            m_pNN->GetLayer(i)->get_delta_all_nodes(temp);

            for(j = 0; j < m_pNN->GetLayer(i)->get_num_nodes(); j++)
            {
                m_pNN->GetLayer(i)->set_node_delta(temp[j], j);
            }
        }
        else //hidden layer
        {
            for(j = 0; j < m_pNN->GetLayer(i)->get_num_nodes(); j++)
            {
                temp[j] = 0.0f;

                for(k = 0; k < m_pNN->GetLayer(i+1)->get_num_nodes(); k++)
                {
                    temp[j] += m_pNN->GetLayer(i+1)->get_node_delta_idx(k) * m_pNN->GetMatrix(i)->get_weight(j, k);
                }
            }
                // temp *= m_pActFunc->apply_act_func_derv(m_ppLys[i]->get_node_value_idx(j));
            m_pNN->GetLayer(i)->get_delta_all_nodes(temp);
            for(j = 0; j < m_pNN->GetLayer(i)->get_num_nodes(); j++)
            {
                m_pNN->GetLayer(i)->set_node_delta(temp[j], j);
            }
        }

        delete [] temp;
        

    }

}