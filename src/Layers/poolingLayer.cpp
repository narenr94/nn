#include "poolingLayer.h"

#include "baseAccelerator.h"
#include <cassert>

#define NOT_APPLICABLE_FOR_POOLING_LAYER assert(0)

PoolingLayer::PoolingLayer(sLayer_Dimensions t_dims, ePooling_type t_pooling_type, ePoolingKernelSize t_kernel_size, uint t_stride):BaseLayer(eLayer_type::POOLING, t_dims, eAct_func::TANH, 0.0f)
{
    m_pooling_type = t_pooling_type;
    m_stride = t_stride;
    m_pooling_kernel_size = t_kernel_size;
}

PoolingLayer::PoolingLayer(std::string load_data):BaseLayer(eLayer_type::DENSE, load_data)
{
}

PoolingLayer::~PoolingLayer()
{
}

void PoolingLayer::do_forwardpass_to_current_layer()
{
    assert(m_bPrevNxtLyrsSet == true);
    m_pAccelerator->do_forwardpass_pooling_layer(m_pooling_type, m_pooling_kernel_size, m_stride);
}

void PoolingLayer::set_transform_matrix_parameter(uint unInIdx, uint unOutIdx, float fWt)
{
    NOT_APPLICABLE_FOR_POOLING_LAYER;
}

void PoolingLayer::set_transform_matrix_parameter(uint Idx, float fWt)
{
    NOT_APPLICABLE_FOR_POOLING_LAYER;
}

void PoolingLayer::set_all_transform_matrix_parameter(float* fWt)
{
    NOT_APPLICABLE_FOR_POOLING_LAYER;
}

std::string PoolingLayer::get_serialized_save_data()
{
    std::ostringstream ss;
    ss << "m_pooling_type: ";
    ss << m_pooling_type << " \n";
    ss << "m_stride: ";
    ss << m_stride << " \n";
    return get_serialized_general_layer_data() + ss.str();
}


