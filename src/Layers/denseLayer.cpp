#include "denseLayer.h"

#include "baseAccelerator.h"
#include <cassert>

DenseLayer::DenseLayer(sLayer_Dimensions t_dims, eAct_func eActFunc, float actParam1):BaseLayer(eLayer_type::DENSE, t_dims, eActFunc, actParam1)
{
}

DenseLayer::DenseLayer(std::string load_data):BaseLayer(eLayer_type::DENSE, load_data)
{
}

DenseLayer::~DenseLayer()
{
}

void DenseLayer::do_forwardpass_to_current_layer()
{
    assert(m_bPrevNxtLyrsSet == true);
    m_pAccelerator->do_forwardpass_dense_layer();
}

void DenseLayer::set_transform_matrix_parameter(uint unInIdx, uint unOutIdx, float fWt)
{
    assert(m_bPrevNxtLyrsSet == true);
    assert((unInIdx < m_pPrevLyr->get_num_nodes()) && (unOutIdx < get_num_nodes()));

    m_pfTransformParameters[(unInIdx * get_num_nodes()) + unOutIdx] = fWt;

}

void DenseLayer::set_transform_matrix_parameter(uint Idx, float fWt)
{
    assert(m_bPrevNxtLyrsSet == true);
    assert(Idx < m_unTransformMatrixSize);
    
    m_pfTransformParameters[Idx] = fWt;

}



void DenseLayer::set_all_transform_matrix_parameter(float* fWt)
{
    assert(m_bPrevNxtLyrsSet == true);
    uint i = 0;
    uint j = 0;
    for(i = 0; i < m_pPrevLyr->get_num_nodes(); i++)
    {
        for(j = 0; j < get_num_nodes(); j++)
        {
            m_pfTransformParameters[(i * get_num_nodes()) + j] = fWt[(i * get_num_nodes()) + j];
        }
    }
}


std::string DenseLayer::get_serialized_save_data()
{
    return get_serialized_general_layer_data() + get_serialized_biases_data() + get_serialized_transform_mtx_data();
}


